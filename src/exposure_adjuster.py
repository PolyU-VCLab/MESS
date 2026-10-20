import torch
import torch.nn as nn

from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
import torch.nn.functional as F

from wavelet_decomp import haar_dwt2d, haar_idwt2d
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


# ---------------------------------------------------------------------------
# Small helper
# ---------------------------------------------------------------------------

def _conv_block(in_ch, out_ch):
    """3x3 ReflectionPad conv + ReLU."""
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=True),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# WaveletFusionNet  (pre-diffusion)
# ---------------------------------------------------------------------------

class WaveletFusionNet(nn.Module):
    """
    Adaptively refines the four Haar subbands (LL, LH, HL, HH) of the input
    image and reconstructs a full-resolution fused image via IDWT, which is
    then fed to the diffusion model.

    Each subband is [B, 3, H/2, W/2].  The network predicts per-subband
    residual corrections; the corrected subbands are then reconstructed back
    to [B, 3, H, W] via haar_idwt2d, preserving high-frequency edge/texture
    structure that bilinear upsampling would otherwise blur.

    Architecture
    ------------
    - Shared encoder : 4 conv blocks on the 12-channel concat of all subbands.
    - Residual heads : four independent 1x1 conv heads (one per subband),
                       zero-initialised so the network starts as identity.
    - Output         : haar_idwt2d(LL + dLL, LH + dLH, HL + dHL, HH + dHH)
                       clamped to [0, 1].
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        # Input: LL, LH+0.5, HL+0.5, HH+0.5 concatenated -> 12 channels
        self.encoder = nn.Sequential(
            _conv_block(12, hidden),
            _conv_block(hidden, hidden),
            _conv_block(hidden, hidden),
            _conv_block(hidden, hidden),
        )
        # Per-subband residual heads: predict additive correction for each subband
        self.head_LL = nn.Conv2d(hidden, 3, kernel_size=1, bias=True)
        self.head_LH = nn.Conv2d(hidden, 3, kernel_size=1, bias=True)
        self.head_HL = nn.Conv2d(hidden, 3, kernel_size=1, bias=True)
        self.head_HH = nn.Conv2d(hidden, 3, kernel_size=1, bias=True)
        # Zero-init so the network starts as identity (no modification)
        for head in [self.head_LL, self.head_LH, self.head_HL, self.head_HH]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, LL, LH, HL, HH, orig_size):
        """
        Args:
            LL, LH, HL, HH : [B, 3, H/2, W/2]  Haar subbands
                             LL in [0, 1]; LH/HL/HH in [-0.5, 0.5]
            orig_size       : (H, W)  kept for API compatibility;
                             IDWT automatically restores original resolution
        Returns:
            fused: [B, 3, H, W] in [0, 1]
        """
        # Shift high-freq subbands to [0, 1] for stable processing
        x = torch.cat([LL, LH + 0.5, HL + 0.5, HH + 0.5], dim=1)  # [B, 12, h, w]
        feat = self.encoder(x)                                        # [B, hidden, h, w]

        # Predict per-subband residuals and apply them
        LL_out = LL + self.head_LL(feat)                              # [B, 3, h, w]
        LH_out = LH + self.head_LH(feat)                              # [B, 3, h, w]
        HL_out = HL + self.head_HL(feat)                              # [B, 3, h, w]
        HH_out = HH + self.head_HH(feat)                              # [B, 3, h, w]

        # Reconstruct full-resolution image via IDWT (preserves frequency structure)
        fused = haar_idwt2d(LL_out, LH_out, HL_out, HH_out)          # [B, 3, H, W]
        return fused.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# ToneMappingNet  (post-diffusion, wavelet-domain, conditioned on ev_norm)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ToneMappingNet  (post-diffusion, wavelet-domain, conditioned on ev_norm)
# ---------------------------------------------------------------------------

class ToneMappingNet(nn.Module):
    """
    Tone-mapping network that operates in the wavelet domain.
    It first fuses the diffusion output with the original input image
    (both decomposed via Haar DWT), then applies FiLM-conditioned
    affine transforms on the fused subbands.

    Pipeline
    --------
    1. Receive diff_out_01 and c_t_01, both in [0, 1].
    2. Apply one level of Haar DWT to each -> 4 subbands each.
    3. A learned, ev_norm-conditioned blending weight alpha fuses the
       two sets of subbands:
           fused_sb = alpha * diff_sb + (1 - alpha) * input_sb
       where alpha is predicted per-subband-group by a small MLP.
    4. A FiLM-conditioned encoder processes the fused subbands and
       predicts per-subband residuals.
    5. Apply IDWT to reconstruct the tone-mapped image.
    6. A final FiLM output gate applies learned per-channel scale + bias.

    Conditioning
    ------------
    ev_norm in [0, 1] is the EV value linearly normalised from [-3, +3].
    """

    def __init__(self, hidden: int = 64, cond_dim: int = 64):
        super().__init__()
        self.cond_dim = cond_dim

        # Condition embedding: scalar ev_norm -> cond vector
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
        )

        # Blending weight predictor: ev_norm -> alpha (one weight per subband)
        # alpha=1 means fully use diffusion output; alpha=0 means fully use input
        self.blend_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim // 2, 4),   # 4 weights: alpha_LL, alpha_LH, alpha_HL, alpha_HH
            nn.Sigmoid(),                   # constrain to [0, 1]
        )
        # Init blend weights to 0.5 (equal mix) at start
        nn.init.zeros_(self.blend_mlp[-2].weight)
        nn.init.zeros_(self.blend_mlp[-2].bias)

        # Shared encoder over fused subbands (12 channels: 4 subbands x 3 channels)
        self.enc1 = _conv_block(12, hidden)
        self.enc2 = _conv_block(hidden, hidden)
        self.enc3 = _conv_block(hidden, hidden)

        # FiLM scale/shift per encoder layer
        self.film1 = nn.Linear(cond_dim, hidden * 2)
        self.film2 = nn.Linear(cond_dim, hidden * 2)
        self.film3 = nn.Linear(cond_dim, hidden * 2)

        # Residual prediction heads: one per subband, 3 output channels each
        self.head_LL = nn.Sequential(_conv_block(hidden, hidden), nn.Conv2d(hidden, 3, 1))
        self.head_LH = nn.Sequential(_conv_block(hidden, hidden), nn.Conv2d(hidden, 3, 1))
        self.head_HL = nn.Sequential(_conv_block(hidden, hidden), nn.Conv2d(hidden, 3, 1))
        self.head_HH = nn.Sequential(_conv_block(hidden, hidden), nn.Conv2d(hidden, 3, 1))

        # Zero-init residual heads so the network starts as identity
        for head in [self.head_LL, self.head_LH, self.head_HL, self.head_HH]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        # FiLM output gate: predicts per-channel (scale, bias) in image space
        self.out_film = nn.Linear(cond_dim, 3 * 2)   # -> [B, 6]: scale(3) + bias(3)
        nn.init.zeros_(self.out_film.weight)
        nn.init.zeros_(self.out_film.bias)

    @staticmethod
    def _film(feat, gamma_beta):
        """FiLM modulation.  feat: [B, C, H, W],  gamma_beta: [B, 2C]."""
        C = feat.shape[1]
        gamma = gamma_beta[:, :C].view(-1, C, 1, 1) + 1.0  # init near 1
        beta  = gamma_beta[:, C:].view(-1, C, 1, 1)
        return feat * gamma + beta

    def forward(self, diff_out_01, c_t_01, ev_norm):
        """
        Args:
            diff_out_01: [B, 3, H, W] in [0, 1]  -- diffusion output
            c_t_01:      [B, 3, H, W] in [0, 1]  -- original input image
            ev_norm:     [B, 1]       in [0, 1]  -- normalised EV condition
        Returns:
            out: [B, 3, H, W] in [0, 1]  -- tone-mapped image
        """
        B = diff_out_01.shape[0]

        # 1) Wavelet decomposition of both images
        LL_d, LH_d, HL_d, HH_d = haar_dwt2d(diff_out_01)   # diffusion subbands
        LL_i, LH_i, HL_i, HH_i = haar_dwt2d(c_t_01)        # input image subbands

        # 2) Condition embedding
        cond = self.cond_mlp(ev_norm.view(B, 1))  # [B, cond_dim]

        # 3) Predict per-subband blending weights from ev_norm
        #    alpha[b, k] in [0, 1]: weight for diffusion output on subband k
        alpha = self.blend_mlp(cond)              # [B, 4]
        a_LL = alpha[:, 0].view(B, 1, 1, 1)      # [B, 1, 1, 1]
        a_LH = alpha[:, 1].view(B, 1, 1, 1)
        a_HL = alpha[:, 2].view(B, 1, 1, 1)
        a_HH = alpha[:, 3].view(B, 1, 1, 1)

        # 4) Fuse subbands: alpha * diff + (1 - alpha) * input
        LL = a_LL * LL_d + (1.0 - a_LL) * LL_i
        LH = a_LH * LH_d + (1.0 - a_LH) * LH_i
        HL = a_HL * HL_d + (1.0 - a_HL) * HL_i
        HH = a_HH * HH_d + (1.0 - a_HH) * HH_i

        # 5) Encode fused subbands with FiLM conditioning
        # Shift high-freq subbands to [0, 1] for stable processing
        x = torch.cat([LL, LH + 0.5, HL + 0.5, HH + 0.5], dim=1)  # [B, 12, h, w]

        f = self._film(self.enc1(x),  self.film1(cond))
        f = self._film(self.enc2(f),  self.film2(cond))
        f = self._film(self.enc3(f),  self.film3(cond))

        # 6) Predict residuals and update each fused subband
        LL_out = LL + self.head_LL(f)
        LH_out = LH + self.head_LH(f)
        HL_out = HL + self.head_HL(f)
        HH_out = HH + self.head_HH(f)

        # 7) Reconstruct via IDWT
        out = haar_idwt2d(LL_out, LH_out, HL_out, HH_out).clamp(0.0, 1.0)

        # 8) FiLM output gate: learned per-channel scale + bias driven by ev_norm
        gb = self.out_film(cond)                          # [B, 6]
        gamma = gb[:, :3].view(B, 3, 1, 1) + 1.0         # init near 1
        beta  = gb[:, 3:].view(B, 3, 1, 1)               # init near 0
        out = (out * gamma + beta).clamp(0.0, 1.0)

        return out

# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ExposureAdjuster(nn.Module):
    """
    Wavelet-diffusion-tonemapping model for EV-guided exposure adjustment.

    Input:
        c_t:      [B, 3, H, W] in [0, 1]   -- input image
        ev_norm:  [B, 1]       in [0, 1]   -- normalised EV (from [-3,+3] to [0,1])

    Pipeline:
        1) Haar DWT of c_t -> LL, LH, HL, HH  (each [B, 3, H/2, W/2])
        2) WaveletFusionNet: adaptive subband combination
           -> fused [B, 3, H, W] in [0, 1]
        3) Diffusion (VAE + 1-step UNet):
           fused [0, 1] -> diff_out [-1, 1]
        4) diff_out mapped to [0, 1]
        5) ToneMappingNet(ev_norm): wavelet-domain tone mapping with FiLM output gate
           -> out [0, 1]
        6) Convert to [-1, 1] output

    Output:
        out: [B, 3, H, W] in [-1, 1]
    """

    def __init__(
        self,
        pretrained_name=None,
        pretrained_path=None,
        lora_rank_unet=8,
        lora_rank_vae=4,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae

        # Lightweight learnable modules
        self.extra_modules = nn.ModuleDict({
            "wavelet_fusion": WaveletFusionNet(hidden=64),
            "tone_mapping":   ToneMappingNet(hidden=64, cond_dim=64),
        }).to(device)

        self.extra_params = nn.ParameterDict({
            "learnable_prompt": nn.Parameter(torch.randn(1, 77, 1024, device=device) * 0.02),
        })

        # 1-step DDPM scheduler
        self.sched = make_1step_sched()

        # VAE with skip connections
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(
            self.vae.encoder, self.vae.encoder.__class__
        )
        self.vae.decoder.forward = my_vae_decoder_fwd.__get__(
            self.vae.decoder, self.vae.decoder.__class__
        )
        self.vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.vae.decoder.ignore_skip = False
        self.vae.to(device)

        # UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="unet"
        ).to(device)

        self.target_modules_vae = [
            "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]
        self.target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0",
            "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj",
        ]

        if pretrained_path is not None:
            self.load_model(pretrained_path)
        elif pretrained_name is None:
            # Fresh init: attach LoRA and zero-init skip convs
            self._attach_lora()
            for conv in [
                self.vae.decoder.skip_conv_1,
                self.vae.decoder.skip_conv_2,
                self.vae.decoder.skip_conv_3,
                self.vae.decoder.skip_conv_4,
            ]:
                nn.init.constant_(conv.weight, 1e-5)
        else:
            raise ValueError(
                "Unsupported init mode. Pass pretrained_path for a checkpoint, "
                "or omit it for fresh init."
            )

        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device=device).long()

    # ------------------------------------------------------------------
    # Property aliases for readability
    # ------------------------------------------------------------------
    @property
    def wavelet_fusion(self):
        return self.extra_modules["wavelet_fusion"]

    @property
    def tone_mapping(self):
        return self.extra_modules["tone_mapping"]

    @property
    def learnable_prompt(self):
        return self.extra_params["learnable_prompt"]

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    def _attach_lora(self):
        self.vae.add_adapter(
            LoraConfig(
                r=self.lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_vae,
            ),
            adapter_name="vae_skip",
        )
        self.unet.add_adapter(
            LoraConfig(
                r=self.lora_rank_unet,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_unet,
            )
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save_model(self, path):
        sd = {
            "model_version": "wavelet_diffusion_tonemapping_v2",
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules":  self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae":  self.lora_rank_vae,
            "state_dict_unet": self.unet.state_dict(),
            "state_dict_vae": {
                k: v for k, v in self.vae.state_dict().items()
                if "lora" in k or "skip" in k
            },
            "extra_modules": self.extra_modules.state_dict(),
            "extra_params":  self.extra_params.state_dict(),
        }
        torch.save(sd, path)

    def load_model(self, path):
        sd = torch.load(path, map_location="cpu")

        if sd.get("model_version") != "wavelet_diffusion_tonemapping_v2":
            raise RuntimeError(
                f"Checkpoint version mismatch: got '{sd.get('model_version')}', "
                f"expected 'wavelet_diffusion_tonemapping_v2'."
            )

        self.target_modules_unet = sd["unet_lora_target_modules"]
        self.target_modules_vae  = sd["vae_lora_target_modules"]
        self.lora_rank_unet = sd["rank_unet"]
        self.lora_rank_vae  = sd["rank_vae"]

        self._attach_lora()

        self.vae.load_state_dict({**self.vae.state_dict(), **sd["state_dict_vae"]})
        self.unet.load_state_dict({**self.unet.state_dict(), **sd["state_dict_unet"]})
        self.extra_modules.load_state_dict(sd["extra_modules"], strict=True)
        self.extra_params.load_state_dict(sd["extra_params"],   strict=True)

        self.to(self.device)

    # ------------------------------------------------------------------
    # Train / Eval modes
    # ------------------------------------------------------------------
    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.extra_modules.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.extra_modules.requires_grad_(False)
        self.extra_params.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.extra_modules.train()

        self.unet.requires_grad_(True)

        # VAE: freeze base weights, train LoRA + skip convs
        self.vae.requires_grad_(False)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

        self.extra_modules.requires_grad_(True)
        self.extra_params.requires_grad_(True)

    # ------------------------------------------------------------------
    # Wavelet fusion  (pre-diffusion)
    # ------------------------------------------------------------------
    def get_fused(self, c_t):
        """
        Decompose c_t with one level of Haar DWT, adaptively refine each
        subband via residual prediction, then reconstruct via IDWT to a
        single [B, 3, H, W] image in [0, 1] that is fed to the diffusion model.

        Args:
            c_t: [B, 3, H, W] in [0, 1]
        Returns:
            fused: [B, 3, H, W] in [0, 1]
        """
        H, W = c_t.shape[2], c_t.shape[3]
        LL, LH, HL, HH = haar_dwt2d(c_t)  # each [B, 3, H/2, W/2]
        # orig_size kept for API compatibility; IDWT internally restores (H, W)
        return self.wavelet_fusion(LL, LH, HL, HH, orig_size=(H, W))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, c_t, ev_norm, deterministic=True):
        """
        Args:
            c_t:           [B, 3, H, W] in [0, 1]
            ev_norm:       [B, 1]       in [0, 1]   -- normalised EV condition
            deterministic: unused (kept for API compatibility)

        Returns:
            out: [B, 3, H, W] in [-1, 1]
        """
        if ev_norm.dim() == 1:
            ev_norm = ev_norm.unsqueeze(1)
        ev_norm = ev_norm.to(device=c_t.device, dtype=c_t.dtype)

        # 1) Wavelet decomposition + adaptive fusion -> diffusion input [0, 1]
        fused = self.get_fused(c_t)  # [B, 3, H, W] in [0, 1]

        # 2) Diffusion: input [0, 1], output [-1, 1]
        B = c_t.shape[0]
        caption_enc = self.learnable_prompt.expand(B, -1, -1).to(dtype=c_t.dtype)

        c_latent = self.vae.encode(fused).latent_dist.sample() * self.vae.config.scaling_factor

        model_pred = self.unet(
            c_latent, self.timesteps, encoder_hidden_states=caption_enc
        ).sample
        x_denoised = self.sched.step(
            model_pred, self.timesteps, c_latent, return_dict=True
        ).prev_sample

        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        diff_out = self.vae.decode(
            x_denoised / self.vae.config.scaling_factor
        ).sample.clamp(-1.0, 1.0)  # [-1, 1]

        # 3) Map diffusion output to [0, 1] for tone mapping
        diff_out_01 = (diff_out + 1.0) * 0.5

        # 4) Wavelet-domain tone mapping: fuse input + diffusion, then affine transform
        c_t_01 = c_t  # c_t is already in [0, 1]
        out_01 = self.tone_mapping(diff_out_01, c_t_01, ev_norm)
        return torch.clamp(out_01 * 2.0 - 1.0, -1.0, 1.0)