# MESS 训练说明（与论文对齐版）

新增文件：

| 文件 | 作用 |
|---|---|
| `src/train_mess.py` | 新的训练入口（按论文 Sec. 3–4 与附录 A–B） |
| `src/my_utils/mess_dataset.py` | 三元组数据集 `{I_i, I_j, I_t*}` + `e_t` |
| `tools/build_mess_manifest.py` | 扫描渲染好的目录，生成 JSONL manifest |
| `scripts/train_mess.sh` | 8 卡启动脚本（论文默认超参） |

网络结构 `src/exposure_adjuster.py` 没有改动；`inference.py` 可以直接读取新脚本保存的 `latest.pkl`。
原有的 `train.py` / `training_utils.py` 保留未动（`inference.py` 仍会 import 里面的 `MEFtoMESDataset`）。

## 1. 论文与原工程训练部分的差异，以及新脚本的处理

| # | 论文 | 原 `train.py` | `train_mess.py` |
|---|---|---|---|
| 1 | 训练样本是三元组 `{I_i, I_j, I_t*}`：同一 RAW 的两张风格渲染，加上一张相机一致的目标曝光渲染 | 二元组 `(MEF, MES[ev])`，输入只用 MEF pseudoGT | 三元组；`I_i, I_j` 从风格池（MEF / TM / 直接渲染）里随机取两张不同的 |
| 2 | `L = λ·‖D(I_i)−D(I_j)‖₁ + ‖Î_t−I_t*‖²₂`，λ=1 | 只有 MSE，没有 `L_con` | 两项都有，`D = get_fused`（WaveletFusionNet） |
| 3 | 在 [0,1] 图像域计算 loss | MSE 在 [-1,1] 域，数值是 [0,1] 域的 4 倍，λ 的相对权重会因此偏移 | 在 [0,1] 域计算 |
| 4 | VAE 冻结；UNet 只训 LoRA + pseudo-word embedding | `unet.requires_grad_(True)` 训练整个 UNet，同时训练 VAE LoRA 和 skip conv | 默认 UNet 只训 LoRA、VAE 完全冻结；如需旧行为，用 `--train_unet_full` / `--train_vae_adapters` 打开 |
| 5 | AdamW(0.9, 0.999)，lr 1e-4，cosine 衰减到 1e-6 | lr 5e-6，constant 调度 | 自定义 cosine 调度，降到 `--min_learning_rate` |
| 6 | 200K iter，总 batch 64（8×A100） | 10K iter，batch 4；同时受 `num_training_epochs=10` 限制，可能提前结束；循环里也不判断 `max_train_steps` | 按迭代数训练（无限 dataloader）；`--total_batch_size 64` 自动推出梯度累积步数 |
| 7 | 512×512 随机裁剪 | `resize_256`（整图缩放，破坏原始像素细节） | 三张图使用同一个随机 512 crop，并做同步水平翻转 |
| 8 | EV 在 [-3,3] 均匀采样，`e_t` 归一化到 [0,1] | 同样的离散 7 档 | 在该场景已有的 EV 中均匀采样（支持非整数 EV），归一化方式与 `inference.py` 一致 |
| 9 | — | 多卡时 `device="cuda"` 写死，所有 rank 都会落到 cuda:0；`timesteps`、scheduler 等非参数张量不会随 `.to()` 迁移 | 构建模型前对每个 rank 执行 `torch.cuda.set_device(local_rank)` |
| 10 | — | 只保存权重 | `latest.pkl`（推理可用）+ `train_state.pt`（optimizer / lr / step），支持 `--resume` |

## 2. 数据提供方式

RAW 渲染部分假定已离线完成，每个 RAW 场景对应一个文件夹：

```
<DATA_ROOT>/<subset>/<scene>/
    style/       # T_sty(X; s)：MEF、色调映射、不同 ISP 风格等，数量不限（≥1）
        mef_mertens.png  tm_reinhard.png  ...
    exposure/    # T_cam(X; e)：固定 ISP、只改变曝光的渲染
        ev-3.png ev-2.png ev-1.png ev+0.png ev+1.png ev+2.png ev+3.png
```

- 同一场景下的所有渲染需要像素对齐；支持 8-bit 和 16-bit 的 PNG/TIFF/JPG。
- EV 从文件名解析：`ev+1.5.png`、`ev_-2.jpg` 和 `-1.png` 都能识别。
- `--style_include_ev 1.0`（默认值）会把 |EV| ≤ 1 的直接渲染也加入风格池，对应论文中 “In addition to directly rendered exposure variants”。设为 `-1` 即关闭。

生成 manifest（JSONL 格式，一行对应一个场景，可以手写，也可以拼接多个）：

```bash
python tools/build_mess_manifest.py --data_root $DATA_ROOT --out $DATA_ROOT/train.jsonl
```

单行格式如下：

```json
{"id":"fivek/a0001","style":["fivek/a0001/style/mef.png"],"exposure":[{"ev":-3.0,"path":"fivek/a0001/exposure/ev-3.png"}]}
```

## 3. 训练

```bash
# 论文设置：8 卡 × 8 张 = 64
DATA_ROOT=/path/to/MESS_RAW bash scripts/train_mess.sh

# 单卡调试：通过梯度累积凑出等效 batch 64
python src/train_mess.py --train_manifest $DATA_ROOT/train.jsonl --output_dir output/mess \
    --train_batch_size 2 --total_batch_size 64 --mixed_precision bf16 --gradient_checkpointing

# 断点续训
python src/train_mess.py ... --resume output/mess/<run>/checkpoints
```

输出目录 `output/mess/<run>/` 包含：
- `checkpoints/latest.pkl`：可直接传给 `src/inference.py --model_path`
- `log/loss.csv`：记录 loss / L_syn / L_con / lr
- `viz/*.jpg`：每行依次为 `I_i | I_j | C_i | Î_t | I_t*`

## 4. 网络结构与论文的不一致（按要求未修改，仅列出）

- **P 的输入**：`ToneMappingNet` 额外接收原图 `I_i`，在小波域与扩散输出融合；论文写的是 `P(I_m, e_t)`。
- **D 的结构**：实现为 “12 通道子带 + 4 层 conv + 残差头”；论文是 “encoder(64ch, 4 ResBlock) → DWT → 通道注意力 → IDWT → decoder + sigmoid”。
- **P 的 FiLM**：实现为多层 FiLM + 混合系数 + 输出门；论文是 “64ch 编码 → DWT → GAP/GMP(256+256) 拼接 EV 嵌入(128) → 640 → γ, β(256)”。
- **pseudo-word**：实现为 77×1024 的可学习 prompt；论文是单个 token。
- **LoRA 位置**：UNet LoRA 覆盖了 conv / ff / proj；论文只在 q/k/v/out。VAE 中额外加了 LoRA 和 skip conv（训练时冻结即可与论文一致）。
- **激活函数**：实现用 ReLU；论文写的是 LeakyReLU(0.2)。
