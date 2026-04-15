import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from diffu_model import load_ddm, shift_logits, get_anneal_attn_mask, get_confidence_entropy
from compress_image_diffugpt import CompressionContext
import constants


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='../Dataset/CIFAR10/cifar10_test/image_24.png')
    parser.add_argument("--model_name", type=str, default='diffugpt-s')
    parser.add_argument("--base_model_name", type=str, default='gpt2')
    parser.add_argument("--model_path", type=str, default='../Model')
    parser.add_argument("--diffu_steps", type=int, default=100)

    # 可视化模式:
    # init  : 不做 rollout，直接看全 MASK 初始状态
    # mid   : rollout 到 T/2 后再可视化
    # final : rollout 到最后一步前再可视化
    # custom: rollout 到 --rollout_until_t 指定的 t 后再可视化
    parser.add_argument(
        "--visualize_mode",
        type=str,
        default="mid",
        choices=["init", "mid", "final", "custom"]
    )
    parser.add_argument(
        "--rollout_until_t",
        type=int,
        default=None,
        help="仅在 visualize_mode=custom 时生效。表示 rollout 停在 t（包含该状态，不再继续 reveal）"
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="confidence_spatial_heatmap_rollout.png")
    parser.add_argument("--checkpoint_path", type=str, default="../Model/diffugpt-s/ddm-sft/train_20251226_231454/checkpoint-26000")
    # parser.add_argument("--checkpoint_path", type=str, default="../Model/diffugpt-s/ddm-sft/train_full_20260415_021825/checkpoint-29000")
    return parser.parse_args()


def get_stop_t(args):
    T = args.diffu_steps
    if args.visualize_mode == "init":
        # 不做 rollout，直接看初始状态
        return T
    if args.visualize_mode == "mid":
        return T // 2
    if args.visualize_mode == "final":
        return 0
    if args.visualize_mode == "custom":
        if args.rollout_until_t is None:
            raise ValueError("visualize_mode=custom 时必须提供 --rollout_until_t")
        if not (0 <= args.rollout_until_t <= T):
            raise ValueError(f"rollout_until_t 必须在 [0, {T}] 范围内")
        return args.rollout_until_t
    raise ValueError(f"Unknown visualize_mode: {args.visualize_mode}")


def rollout_to_state(
    x_true_ids: torch.Tensor,
    tokenizer,
    model,
    ctx,
    attention_mask: torch.Tensor,
    diffu_steps: int,
    stop_t: int,
):
    """
    从全 MASK 状态开始，按压缩端一致的高置信度 reveal 逻辑 rollout，
    一直运行到 t = stop_t 之前为止，返回当前 xt 和 maskable_mask。

    约定：
    - 若 stop_t == diffu_steps，则不做任何 rollout，直接返回全 MASK 初始状态
    - 循环执行 t = diffu_steps-1, ..., stop_t
      如果想“rollout 到 T/2 后可视化”，就让 stop_t = T//2
    """
    device = x_true_ids.device
    batch_size, seq_len = x_true_ids.shape

    xt = torch.full((batch_size, seq_len), tokenizer.mask_token_id, dtype=torch.long, device=device)
    xt[:, 0] = tokenizer.bos_token_id

    maskable_mask = torch.ones_like(xt, dtype=torch.bool, device=device)
    maskable_mask[:, 0] = False

    if stop_t >= diffu_steps:
        return xt, maskable_mask

    with torch.no_grad():
        for t in range(diffu_steps - 1, stop_t - 1, -1):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                raw_logits = model(xt, attention_mask=attention_mask)

            logits_shifted = shift_logits(raw_logits)
            logits_pixel = logits_shifted[:, :, ctx.pixel_token_ids]

            confidences = get_confidence_entropy(logits_pixel, None)
            confidences = confidences.masked_fill(~maskable_mask, float("-inf"))
            confidences = torch.round(confidences * 1e4) / 1e4

            masks_left_per_sample = torch.sum(maskable_mask, dim=1)
            num_current_masks = masks_left_per_sample.max().item()
            if num_current_masks == 0:
                break

            ratio = 1.0 / (t + 1)
            k = max(1, min(int(num_current_masks * ratio), num_current_masks))

            sorted_indices = torch.argsort(confidences, dim=-1, descending=True, stable=True)
            target_indices = sorted_indices[:, :k]
            true_token_ids = torch.gather(x_true_ids, 1, target_indices)

            arange_k = torch.arange(k, device=device).unsqueeze(0).expand(batch_size, -1)
            valid_k_mask = arange_k < masks_left_per_sample.unsqueeze(1)

            flat_indices = target_indices.reshape(-1)
            flat_src = true_token_ids.reshape(-1)
            flat_mask = valid_k_mask.reshape(-1)

            batch_offsets = torch.arange(batch_size, device=device) * seq_len
            batch_offsets = batch_offsets.unsqueeze(1).expand(-1, k).reshape(-1)
            final_flat_indices = batch_offsets + flat_indices

            xt.view(-1)[final_flat_indices[flat_mask]] = flat_src[flat_mask]
            maskable_mask.view(-1)[final_flat_indices[flat_mask]] = False

    return xt, maskable_mask


def main(args):
    print("🚀 正在加载模型与 Tokenizer...")
    args.model_path = args.checkpoint_path
    tokenizer, model = load_ddm(args)
    model.eval()
    device = model.device
    ctx = CompressionContext(tokenizer)

    print(f"🖼️ 正在读取图像: {args.image_path}")
    img = Image.open(args.image_path).convert('RGB')
    img_np = np.array(img)

    H, W, C = img_np.shape
    patch_h, patch_w = constants.CHUNK_SHAPE_2D

    # 裁剪到 patch 整数倍
    new_H = (H // patch_h) * patch_h
    new_W = (W // patch_w) * patch_w
    img_np = img_np[:new_H, :new_W, :]
    Nh = new_H // patch_h
    Nw = new_W // patch_w

    print(f"裁剪后图像尺寸: {new_H}x{new_W}, 共有 {Nh * Nw} 个 Patch")

    # 图像分块
    patches = img_np.reshape(Nh, patch_h, Nw, patch_w, C).transpose(0, 2, 1, 3, 4)
    patches_flat = patches.reshape(Nh * Nw, -1)

    batch_size = args.batch_size
    all_patch_confs = []
    all_patch_mask_states = []

    seq_len = patch_h * patch_w * C + 1
    stop_t = get_stop_t(args)

    if args.visualize_mode == "init":
        print("⚡ 可视化模式: init（全 MASK 初始状态）")
    else:
        print(f"⚡ 可视化模式: {args.visualize_mode}，rollout 到 t={stop_t} 后再可视化")

    with torch.inference_mode():
        for i in tqdm(range(0, len(patches_flat), batch_size)):
            patch_chunk = patches_flat[i:i + batch_size]
            bsz = len(patch_chunk)

            true_pixels = torch.tensor(patch_chunk, dtype=torch.long, device=device)
            true_ids_body = ctx.pixel_to_token_tensor[true_pixels]
            x_true_ids = torch.cat(
                [torch.full((bsz, 1), tokenizer.bos_token_id, dtype=torch.long, device=device), true_ids_body],
                dim=1
            )

            attention_mask = get_anneal_attn_mask(
                seq_len,
                bsz,
                dtype=model.denoise_model.dtype,
                device=device,
                attn_mask_ratio=1.0
            )

            # 先 rollout 到目标中间状态
            xt, maskable_mask = rollout_to_state(
                x_true_ids=x_true_ids,
                tokenizer=tokenizer,
                model=model,
                ctx=ctx,
                attention_mask=attention_mask,
                diffu_steps=args.diffu_steps,
                stop_t=stop_t,
            )

            # 在当前中间状态重新计算一次置信度
            with torch.autocast(device_type=device, dtype=torch.float16):
                raw_logits = model(xt, attention_mask=attention_mask)

            logits_shifted = shift_logits(raw_logits)
            logits_pixel = logits_shifted[:, :, ctx.pixel_token_ids]
            confidences = get_confidence_entropy(logits_pixel, None)

            # 只保留像素位置，去掉 BOS
            patch_conf = abs(confidences[:, 1:].float().cpu().numpy())
            patch_mask = maskable_mask[:, 1:].float().cpu().numpy()

            all_patch_confs.append(patch_conf)
            all_patch_mask_states.append(patch_mask)

    all_patch_confs = np.concatenate(all_patch_confs, axis=0)       # [Nh*Nw, 768]
    all_patch_mask_states = np.concatenate(all_patch_mask_states, axis=0)  # [Nh*Nw, 768]

    conf_patches = all_patch_confs.reshape(Nh, Nw, patch_h, patch_w, C)
    mask_patches = all_patch_mask_states.reshape(Nh, Nw, patch_h, patch_w, C)

    # RGB 通道取均值，得到空间 heatmap
    conf_spatial = conf_patches.mean(axis=-1)   # [Nh, Nw, ph, pw]
    mask_spatial = mask_patches.mean(axis=-1)   # [Nh, Nw, ph, pw]

    heatmap = conf_spatial.transpose(0, 2, 1, 3).reshape(new_H, new_W)
    maskmap = mask_spatial.transpose(0, 2, 1, 3).reshape(new_H, new_W)

    # 只看“当前仍然 mask 的位置”的置信度更有解释性
    masked_heatmap = heatmap.copy()
    masked_heatmap[maskmap < 0.5] = np.nan

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image", fontsize=14)
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Current Mask State (1=Masked, t={stop_t})", fontsize=14)
    plt.imshow(maskmap, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Confidence on Remaining Masks (t={stop_t})", fontsize=14)
    im = plt.imshow(masked_heatmap, cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Entropy")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 已保存到: {args.save_path}")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    args = get_args()
    main(args)