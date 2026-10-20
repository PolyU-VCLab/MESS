import csv
import os
from datetime import datetime
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from exposure_adjuster import ExposureAdjuster
from my_utils.training_utils import parse_args, MEFtoMESDataset


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "log"), exist_ok=True)

    model = ExposureAdjuster(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
    model.set_train()

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        model.unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    layers_to_opt = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    dataset_train = MEFtoMESDataset(
        mes_folder=args.mes_folder,
        mef_folder=args.mef_folder,
        image_prep=args.train_image_prep,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    model, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        model, optimizer, dl_train, lr_scheduler
    )
    model.to(accelerator.device, dtype=torch.float32)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    loss_accum = 0.0
    loss_count = 0
    log_path = os.path.join(run_dir, "log", "loss.csv")

    if accelerator.is_main_process:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "avg_loss"])

    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(model):
                x_src = batch["img1"].to(accelerator.device)
                x_tgt = batch["img2"].to(accelerator.device)
                ev_norm = batch["ev_norm"].to(accelerator.device)

                x_tgt_pred = model(x_src, ev_norm, deterministic=True)
                loss = torch.mean((x_tgt_pred - x_tgt) ** 2)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                loss_accum += loss.item()
                loss_count += 1

                if accelerator.is_main_process:
                    progress_bar.set_postfix(loss_l2=loss.item())

                    if global_step % args.checkpointing_steps == 0:
                        # --- save checkpoint ---
                        ckpt_dir = os.path.join(run_dir, "checkpoints")
                        latest_path = os.path.join(ckpt_dir, "latest.pkl")
                        tmp_path = latest_path + ".tmp"
                        # Atomic write: write to tmp first, then replace
                        accelerator.unwrap_model(model).save_model(tmp_path)
                        os.replace(tmp_path, latest_path)

                        # --- log average loss for this window ---
                        avg_loss = loss_accum / loss_count
                        with open(log_path, "a", newline="") as f:
                            csv.writer(f).writerow([global_step, f"{avg_loss:.6f}"])
                        loss_accum = 0.0
                        loss_count = 0


if __name__ == "__main__":
    args = parse_args()
    main(args)