import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accelerate import Accelerator, DistributedDataParallelKwargs  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402

from exposure_adjuster import ExposureAdjuster  # noqa: E402
from my_utils.mess_dataset import MESSTripletDataset, EV_MIN, EV_MAX  # noqa: E402


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    p = argparse.ArgumentParser(description="MESS training.")

    # Data
    p.add_argument("--train_manifest", nargs="+", required=True, help="JSONL manifest(s) of RAW-guided renders.")
    p.add_argument("--val_manifest", nargs="+", default=None, help="Optional manifest(s) for periodic validation.")
    p.add_argument("--data_root", default=None, help="Root for relative paths (default: manifest dir).")
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--style_include_ev", type=float, default=1.0,
                   help="Add camera-consistent renders with |EV|<=x to the style pool (<0 disables).")
    p.add_argument("--no_hflip", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=8)

    # Output / logging
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default=None, help="Sub-folder name (default: timestamp).")
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--checkpointing_steps", type=int, default=5000)
    p.add_argument("--keep_step_checkpoints", action="store_true",
                   help="Also keep step_XXXXXX.pkl besides latest.pkl.")
    p.add_argument("--viz_steps", type=int, default=2000, help="Save a visual grid every N steps (0 = off).")
    p.add_argument("--val_steps", type=int, default=10000)
    p.add_argument("--val_max_batches", type=int, default=20)
    p.add_argument("--resume", default=None,
                   help="Path to a run's checkpoints dir (or 'latest' inside --output_dir/--run_name).")

    # Model
    p.add_argument("--lora_rank_unet", type=int, default=8)
    p.add_argument("--lora_rank_vae", type=int, default=4)
    p.add_argument("--init_from", default=None, help="Initialise network weights from a .pkl (no optimiser state).")
    p.add_argument("--train_vae_adapters", action="store_true",
                   help="also train VAE LoRA + skip convs.")
    p.add_argument("--train_unet_full", action="store_true",
                   help="train all UNet weights instead of LoRA only.")

    # Loss
    p.add_argument("--lambda_con", type=float, default=1.0, help="Weight of L_con.")

    # Optimisation
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_steps", type=int, default=200_000)
    p.add_argument("--train_batch_size", type=int, default=8, help="Per-GPU micro-batch size.")
    p.add_argument("--total_batch_size", type=int, default=64,
                   help="Effective batch = per_gpu * num_gpus * grad_accum. Grad accumulation is derived "
                        "from it; set <=0 to use --gradient_accumulation_steps directly.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--min_learning_rate", type=float, default=1e-6)
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mixed_precision", default="no", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    args = p.parse_args(input_args)
    assert args.crop_size % 8 == 0, "--crop_size must be a multiple of 8"
    return args


# ---------------------------------------------------------------------------
# Training wrapper: one forward that yields everything the losses need.
# (Keeps DDP happy: all trainable params are touched inside a single forward.)
# ---------------------------------------------------------------------------

class MESSTrainWrapper(nn.Module):
    def __init__(self, net: ExposureAdjuster, use_con: bool):
        super().__init__()
        self.net = net
        self.use_con = use_con

    def forward(self, img_i, img_j, ev_norm):
        # \hat I_t = P(G(D(I_i)), e_t)  -- network output in [-1, 1] -> [0, 1]
        pred_01 = (self.net(img_i, ev_norm) + 1.0) * 0.5
        if self.use_con:
            # C_i = D(I_i), C_j = D(I_j)  (D == WaveletFusionNet via get_fused)
            c = self.net.get_fused(torch.cat([img_i, img_j], dim=0))
            c_i, c_j = c.chunk(2, dim=0)
        else:
            c_i = c_j = None
        return pred_01, c_i, c_j


def configure_trainable(net: ExposureAdjuster, args):
    net.requires_grad_(False)

    # D, P (+ any lightweight extra modules) and the pseudo-word embedding p
    net.extra_modules.requires_grad_(True)
    net.extra_params.requires_grad_(True)

    if args.train_unet_full:
        net.unet.requires_grad_(True)
    else:
        for n, prm in net.unet.named_parameters():
            if "lora_" in n:
                prm.requires_grad_(True)

    if args.train_vae_adapters:
        for n, prm in net.vae.named_parameters():
            if "lora_" in n or "skip_conv" in n:
                prm.requires_grad_(True)

    net.unet.train()
    net.extra_modules.train()
    net.vae.train(args.train_vae_adapters)

    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    return named


def count_params(named):
    groups = {}
    for n, p in named:
        key = n.split(".")[0] if not n.startswith("extra_modules") else ".".join(n.split(".")[:2])
        groups[key] = groups.get(key, 0) + p.numel()
    return groups

def build_lr_lambda(base_lr, min_lr, warmup, total):
    floor = min_lr / base_lr

    def fn(step):
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        t = min(max(step - warmup, 0) / max(total - warmup, 1), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))
    return fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infinite_loader(dl, accelerator):
    epoch = 0
    while True:
        sampler = getattr(dl, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if hasattr(dl, "set_epoch"):
            dl.set_epoch(epoch)
        for batch in dl:
            yield batch
        epoch += 1


@torch.no_grad()
def save_viz(path, batch, pred_01, c_i, n=4):
    import torchvision.utils as vutils
    n = min(n, pred_01.shape[0])
    rows = [batch["img_i"][:n], batch["img_j"][:n]]
    if c_i is not None:
        rows.append(c_i[:n])
    rows += [pred_01[:n], batch["img_t"][:n]]
    # each row: I_i | I_j | C_i | \hat I_t | I_t*
    grid = torch.stack([r.float().clamp(0, 1).cpu() for r in rows], dim=1).flatten(0, 1)
    vutils.save_image(grid, path, nrow=len(rows))


def save_checkpoint(accelerator, wrapper, optimizer, lr_sched, step, ckpt_dir, keep_step):
    if not accelerator.is_main_process:
        return
    net = accelerator.unwrap_model(wrapper).net
    latest = os.path.join(ckpt_dir, "latest.pkl")
    net.save_model(latest + ".tmp")                     # inference-compatible checkpoint
    os.replace(latest + ".tmp", latest)
    if keep_step:
        net.save_model(os.path.join(ckpt_dir, f"step_{step:07d}.pkl"))
    state = {"step": step, "optimizer": optimizer.state_dict(), "lr_sched": lr_sched.state_dict()}
    torch.save(state, os.path.join(ckpt_dir, "train_state.pt.tmp"))
    os.replace(os.path.join(ckpt_dir, "train_state.pt.tmp"), os.path.join(ckpt_dir, "train_state.pt"))


@torch.no_grad()
def run_validation(accelerator, wrapper, dl_val, max_batches):
    wrapper.eval()
    tot = torch.zeros(3, device=accelerator.device)  # mse_sum, psnr_sum, count
    for k, batch in enumerate(dl_val):
        if k >= max_batches:
            break
        pred, _, _ = wrapper(batch["img_i"], batch["img_j"], batch["ev_norm"])
        mse = ((pred.float() - batch["img_t"].float()) ** 2).flatten(1).mean(1)
        tot[0] += mse.sum()
        tot[1] += (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).sum()
        tot[2] += mse.numel()
    tot = accelerator.reduce(tot, reduction="sum")
    wrapper.train()
    # keep frozen VAE in eval mode after toggling
    return (tot[0] / tot[2]).item(), (tot[1] / tot[2]).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.total_batch_size and args.total_batch_size > 0:
        per_step = args.train_batch_size * world
        if args.total_batch_size % per_step != 0:
            raise ValueError(f"total_batch_size={args.total_batch_size} not divisible by "
                             f"train_batch_size*world={per_step}")
        args.gradient_accumulation_steps = args.total_batch_size // per_step

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    # ExposureAdjuster / make_1step_sched hard-code "cuda" -> pin the current device per rank
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)
    set_seed(args.seed + accelerator.process_index)  # different crops/EVs per rank

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------- run dirs ----------------
    if args.resume == "latest" and args.run_name is None:
        raise ValueError("--resume latest requires --run_name to locate <output_dir>/<run_name>/checkpoints")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume not in (None, "latest") and args.run_name is None:
        # continue writing into the run that owns the checkpoint dir
        run_dir = os.path.dirname(os.path.abspath(args.resume.rstrip("/")))
    else:
        run_dir = os.path.join(args.output_dir, run_name)
    ckpt_dir, log_dir, viz_dir = (os.path.join(run_dir, d) for d in ("checkpoints", "log", "viz"))
    if accelerator.is_main_process:
        for d in (ckpt_dir, log_dir, viz_dir):
            os.makedirs(d, exist_ok=True)
        with open(os.path.join(log_dir, "args.txt"), "a") as f:
            f.write(f"# {datetime.now().isoformat()}  world={world}\n{vars(args)}\n")

    # ---------------- model ----------------
    device = accelerator.device
    resume_dir = None
    if args.resume is not None:
        resume_dir = args.resume if args.resume != "latest" else ckpt_dir
        net = ExposureAdjuster(pretrained_path=os.path.join(resume_dir, "latest.pkl"), device=device)
    elif args.init_from is not None:
        net = ExposureAdjuster(pretrained_path=args.init_from, device=device)
    else:
        net = ExposureAdjuster(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, device=device)

    if args.enable_xformers_memory_efficient_attention:
        try:
            net.unet.enable_xformers_memory_efficient_attention()
        except Exception as e:  # pragma: no cover
            accelerator.print(f"[warn] xformers unavailable: {e}")
    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()

    named_trainable = configure_trainable(net, args)
    net.to(device, dtype=torch.float32)  # master weights fp32; autocast handles mixed precision
    if accelerator.is_main_process:
        for k, v in count_params(named_trainable).items():
            accelerator.print(f"  trainable {k:<32s} {v / 1e6:8.3f} M")

    wrapper = MESSTrainWrapper(net, use_con=args.lambda_con > 0)

    # ---------------- optimiser / scheduler ----------------
    params = [p for _, p in named_trainable]
    optimizer = torch.optim.AdamW(
        params, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.learning_rate, args.min_learning_rate,
                                   args.lr_warmup_steps, args.max_train_steps))

    # ---------------- data ----------------
    ds_train = MESSTripletDataset(
        args.train_manifest, data_root=args.data_root, crop_size=args.crop_size,
        ev_min=EV_MIN, ev_max=EV_MAX, style_include_ev=args.style_include_ev, hflip=not args.no_hflip,
    )
    dl_train = DataLoader(
        ds_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
        num_workers=args.dataloader_num_workers, pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    dl_val = None
    if args.val_manifest:
        ds_val = MESSTripletDataset(args.val_manifest, data_root=args.data_root, crop_size=args.crop_size,
                                    style_include_ev=args.style_include_ev, hflip=False)
        dl_val = DataLoader(ds_val, batch_size=args.train_batch_size, shuffle=False,
                            num_workers=min(4, args.dataloader_num_workers))
    accelerator.print(f"train scenes: {len(ds_train)} | per-GPU batch {args.train_batch_size} x "
                      f"{accelerator.num_processes} GPU x accum {args.gradient_accumulation_steps} "
                      f"= {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # LR scheduler is stepped manually once per optimisation step -> not passed to prepare()
    wrapper, optimizer, dl_train = accelerator.prepare(wrapper, optimizer, dl_train)
    if dl_val is not None:
        dl_val = accelerator.prepare(dl_val)

    # ---------------- resume ----------------
    global_step = 0
    if resume_dir is not None:
        st_path = os.path.join(resume_dir, "train_state.pt")
        if os.path.isfile(st_path):
            st = torch.load(st_path, map_location="cpu")
            optimizer.load_state_dict(st["optimizer"])
            lr_sched.load_state_dict(st["lr_sched"])
            global_step = int(st["step"])
            accelerator.print(f"Resumed from {resume_dir} at step {global_step}")
        else:
            accelerator.print(f"[warn] {st_path} not found: weights restored, optimiser restarted.")

    # ---------------- logging ----------------
    log_path = os.path.join(log_dir, "loss.csv")
    if accelerator.is_main_process and not os.path.isfile(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "loss_syn", "loss_con", "lr", "sec_per_step"])

    run_sum = torch.zeros(3, device=device)  # loss, syn, con
    run_cnt = 0
    t0 = time.time()
    data_iter = infinite_loader(dl_train, accelerator)
    wrapper.train()
    net_ref = accelerator.unwrap_model(wrapper).net
    net_ref.vae.train(args.train_vae_adapters)

    if accelerator.is_main_process:
        from tqdm.auto import tqdm
        pbar = tqdm(total=args.max_train_steps, initial=global_step, desc="steps")
    else:
        pbar = None

    while global_step < args.max_train_steps:
        batch = next(data_iter)
        with accelerator.accumulate(wrapper):
            pred_01, c_i, c_j = wrapper(batch["img_i"], batch["img_j"], batch["ev_norm"])

            # L_syn = ||\hat I_t - I_t*||_2^2   (mean over pixels, images in [0, 1])
            loss_syn = F.mse_loss(pred_01.float(), batch["img_t"].float())
            # L_con = ||C_i - C_j||_1
            if c_i is not None:
                loss_con = F.l1_loss(c_i.float(), c_j.float())
            else:
                loss_con = torch.zeros((), device=device)
            loss = loss_syn + args.lambda_con * loss_con

            accelerator.backward(loss)
            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        run_sum += torch.stack([loss.detach(), loss_syn.detach(), loss_con.detach()]).float()
        run_cnt += 1

        if not accelerator.sync_gradients:
            continue

        # ---- one optimisation step finished ----
        lr_sched.step()
        global_step += 1
        if pbar is not None:
            pbar.update(1)

        if global_step % args.log_steps == 0:
            avg = accelerator.reduce(run_sum / max(run_cnt, 1), reduction="mean")
            sps = (time.time() - t0) / args.log_steps
            if accelerator.is_main_process:
                lr = lr_sched.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg[0]:.4f}", syn=f"{avg[1]:.4f}", con=f"{avg[2]:.4f}", lr=f"{lr:.2e}")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([global_step, f"{avg[0]:.6f}", f"{avg[1]:.6f}",
                                            f"{avg[2]:.6f}", f"{lr:.3e}", f"{sps:.3f}"])
            run_sum.zero_()
            run_cnt = 0
            t0 = time.time()

        if args.viz_steps and global_step % args.viz_steps == 0 and accelerator.is_main_process:
            save_viz(os.path.join(viz_dir, f"step_{global_step:07d}.jpg"), batch, pred_01.detach(),
                     None if c_i is None else c_i.detach())

        if dl_val is not None and args.val_steps and global_step % args.val_steps == 0:
            mse, psnr = run_validation(accelerator, wrapper, dl_val, args.val_max_batches)
            net_ref.vae.train(args.train_vae_adapters)
            if accelerator.is_main_process:
                accelerator.print(f"[val] step {global_step}: mse={mse:.5f} psnr={psnr:.2f}dB")
                with open(os.path.join(log_dir, "val.csv"), "a", newline="") as f:
                    csv.writer(f).writerow([global_step, f"{mse:.6f}", f"{psnr:.3f}"])

        if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
            accelerator.wait_for_everyone()
            save_checkpoint(accelerator, wrapper, optimizer, lr_sched, global_step,
                            ckpt_dir, args.keep_step_checkpoints)

    accelerator.wait_for_everyone()
    if pbar is not None:
        pbar.close()
    accelerator.print(f"Done. Final checkpoint: {os.path.join(ckpt_dir, 'latest.pkl')}")
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
