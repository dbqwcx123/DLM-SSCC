import os
# 1. 解决 Tokenizers 死锁警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

from diffu_model import *
import constants
from data_loaders import patch_visualize
from utils import arithmetic_coder
from utils.ac_utils import normalize_pdf_for_arithmetic_coding
from utils.ECCT_utils import set_seed
from utils.pixel_token_dict import *

# ==========================================
# 1. TimeProfiler 类 (保持一致用于性能分析)
# ==========================================
class TimeProfiler:
    def __init__(self, enabled=True):
        self.records = {}
        self.enabled = enabled
        self.current_start = None
        self.current_name = None

    def tick(self, name):
        if not self.enabled: return
        torch.cuda.synchronize()
        now = time.time()
        if self.current_start is not None:
            elapsed = now - self.current_start
            if self.current_name not in self.records:
                self.records[self.current_name] = 0.0
            self.records[self.current_name] += elapsed
        self.current_name = name
        self.current_start = now

    def end_tick(self):
        if not self.enabled or self.current_start is None: return
        torch.cuda.synchronize()
        now = time.time()
        elapsed = now - self.current_start
        if self.current_name not in self.records:
            self.records[self.current_name] = 0.0
        self.records[self.current_name] += elapsed
        self.current_start = None
        self.current_name = None

    def print_stats(self):
        if not self.enabled: return
        print("\n" + "="*40)
        print("⚡ 解码性能耗时分布 (Time Profiling)")
        print("="*40)
        total_time = sum(self.records.values())
        if total_time == 0: return
        sorted_records = sorted(self.records.items(), key=lambda x: x[1], reverse=True)
        for name, duration in sorted_records:
            percentage = (duration / total_time) * 100
            print(f"{name:<25}: {duration:.4f}s ({percentage:.1f}%)")
        print("="*40 + "\n")

profiler = TimeProfiler(enabled=True)

# ==========================================
# 2. 解压缩上下文
# ==========================================
class CompressionContext:
    def __init__(self, tokenizer):
        self.pixel_token_ids = compute_pixel_token_ids(tokenizer)
        self.max_token_id = tokenizer.vocab_size
        
        # [GPU] Token ID -> Pixel Value 映射表
        self.id_to_pixel_tensor = torch.full((self.max_token_id,), -1, dtype=torch.long, device='cuda')
        pixel_vals = torch.arange(len(self.pixel_token_ids), device='cuda')
        token_indices = torch.tensor(self.pixel_token_ids, device='cuda')
        self.id_to_pixel_tensor[token_indices] = pixel_vals
        
        # [GPU] Pixel Value -> Token ID 映射表
        self.pixel_to_token_tensor = token_indices.clone()

def make_input_fn(bit_string):
    """为算术解码器构建输入函数"""
    iterator = iter(bit_string)
    def _fn():
        try:
            return int(next(iterator))
        except StopIteration:
            # 如果流耗尽，通常意味着解码结束或Padding
            return 0 
    return _fn

# ==========================================
# 4. 核心：真·Batch 解压逻辑
# ==========================================
def decompress_image_batched(batch_bit_strings, model, tokenizer, ctx, args):
    """
    batch_bit_strings: List[str], 一个 batch 的比特流字符串列表
    """
    profiler.tick("Setup")
    model.eval()
    device = model.device
    batch_size = len(batch_bit_strings)
    
    # 1. 初始化序列参数
    h, w = constants.CHUNK_SHAPE_2D
    channels = 1 if constants.IS_CHANNEL_WISED else 3
    seq_len = h * w * channels + 1 # +1 for BOS
    
    # 2. 准备初始状态
    # 创建 BOS + 全 Mask 的序列
    # xt: [Batch, Seq_Len]
    xt = torch.full((batch_size, seq_len), tokenizer.mask_token_id, dtype=torch.long, device=device)
    xt[:, 0] = tokenizer.bos_token_id # Set BOS
    
    # Mask 初始化
    # maskable_mask: True 表示该位置是 Mask，需要预测
    maskable_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    maskable_mask[:, 0] = False # BOS 不可预测
    
    # Attention Mask
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=model.denoise_model.dtype, device=device, attn_mask_ratio=1.0)
    
    # 3. 初始化批量算术解码器
    decoders = []
    for i in range(batch_size):
        decoders.append(arithmetic_coder.Decoder(
            base=constants.ARITHMETIC_CODER_BASE,
            precision=constants.ARITHMETIC_CODER_PRECISION,
            input_fn=make_input_fn(batch_bit_strings[i])
        ))
    
    profiler.tick("Model Inference (GPU)")

    with torch.inference_mode():
        for t in range(args.diffusion_steps-1, -1, -1):
            
            # 1. GPU 推理
            # with torch.cuda.amp.autocast(enabled=True):  # 混合精度
            with torch.no_grad():
                raw_logits = model(xt, attention_mask=attention_mask)
            
            # 2. Logits 处理 & Shift
            logits_shifted = shift_logits(raw_logits)
            logits_pixel = logits_shifted[:,:, ctx.pixel_token_ids]
            
            # 3. 置信度计算
            if args.confidence_st == 'entropy':
                confidences = get_confidence_entropy(logits_pixel, None)
            else:
                probs = F.softmax(logits_pixel, dim=-1)
                max_probs, _ = probs.max(dim=-1)
                confidences = max_probs
            
            # 仅关注还未解码的位置 (maskable_mask 为 True 的位置)
            confidences = confidences.masked_fill(~maskable_mask, float('-inf'))
            
            # -----------------------------------------------------------------------------
            # 计算每个样本实际剩余的 mask 数量 [B]
            masks_left_per_sample = torch.sum(maskable_mask, dim=1)
            num_current_masks = masks_left_per_sample.max().item()
            if num_current_masks == 0: break
            
            ratio = 1.0 / (t + 1)
            k = max(1, min(int(num_current_masks * ratio), num_current_masks))
            
            # 排序
            sorted_indices = batch_stable_sort(confidences)
            target_indices = sorted_indices[:, :k]
            
            # 提取概率
            batch_indices_expanded = target_indices.unsqueeze(-1).expand(-1, -1, logits_pixel.size(-1))
            target_logits = torch.gather(logits_pixel, 1, batch_indices_expanded)
            target_probs = torch.softmax(target_logits.double(), dim=-1) 
            
            target_probs_cpu = target_probs.cpu().numpy()
            masks_left_cpu = masks_left_per_sample.cpu().numpy()
            
            profiler.end_tick()
            profiler.tick("Arithmetic Decoding (CPU)")
            
            decoded_pixel_values = np.zeros((batch_size, k), dtype=np.int64)
            
            # 串行解码 (Batch内循环)
            for b in range(batch_size):
                dec = decoders[b]
                p_b = target_probs_cpu[b] 
                
                # 【关键修复】：只解码实际需要的数量
                actual_k = min(k, int(masks_left_cpu[b]))
                
                for i in range(actual_k):
                    try:
                        pixel_val = dec.decode(normalize_pdf_for_arithmetic_coding(p_b[i]))
                        # ... (越界检查代码不变)
                    except Exception as e:
                        # ... (错误处理不变)
                        pixel_val = 128
                    decoded_pixel_values[b, i] = pixel_val
            
            profiler.end_tick()
            profiler.tick("Model Inference (GPU)")
            
            # 7. 更新状态 (GPU)
            decoded_pixels_tensor = torch.tensor(decoded_pixel_values, device=device, dtype=torch.long)
            decoded_token_ids = ctx.pixel_to_token_tensor[decoded_pixels_tensor] # [B, K]
            
            # 【关键修复】：构建有效性 Mask 并只更新有效位置
            arange_k = torch.arange(k, device=device).unsqueeze(0).expand(batch_size, -1)
            masks_left_expanded = masks_left_per_sample.unsqueeze(1).expand(-1, k)
            valid_k_mask = arange_k < masks_left_expanded # [B, K]
            
            # 展平准备 Scatter
            flat_indices = target_indices.reshape(-1)
            flat_src = decoded_token_ids.reshape(-1)
            flat_mask = valid_k_mask.reshape(-1)
            
            # 计算在整个 xt (flattened) 中的绝对索引
            batch_offsets = torch.arange(batch_size, device=device) * xt.size(1)
            batch_offsets = batch_offsets.unsqueeze(1).expand(-1, k).reshape(-1)
            final_flat_indices = batch_offsets + flat_indices
            
            # 只更新 valid 的部分！
            # 如果不加这个 mask，decoded_pixel_values 里的 0 (初始值) 会覆盖掉那些
            # 本来已经解码完成（但因为 batch 同步被强制选中的）像素。
            xt.view(-1)[final_flat_indices[flat_mask]] = flat_src[flat_mask]
            
            # 更新 Mask
            maskable_mask.view(-1)[final_flat_indices[flat_mask]] = False
            
            # # -----------------------------------------------------------------------------
            # # 计算当前步需要解码的 token 数量 k
            # num_current_masks = torch.sum(maskable_mask, dim=1).max().item()
            # if num_current_masks == 0: break
            
            # ratio = 1.0 / (t + 1)
            # k = max(1, min(int(num_current_masks * ratio), num_current_masks))
            
            # # 4. 使用稳定排序策略获取解码位置
            # # _, sorted_indices = torch.sort(confidences, descending=True, stable=True)
            # sorted_indices = batch_stable_sort(confidences)
            # target_indices = sorted_indices[:, :k]
            
            # # 5. 提取这些位置的概率分布
            # batch_indices_expanded = target_indices.unsqueeze(-1).expand(-1, -1, logits_pixel.size(-1))
            # target_logits = torch.gather(logits_pixel, 1, batch_indices_expanded)
            # target_probs = torch.softmax(target_logits.double(), dim=-1) # [B, K, 256]
            
            # # 6. 数据回传 CPU 进行解码
            # target_probs_cpu = target_probs.cpu().numpy()
            
            # profiler.end_tick()
            # profiler.tick("Arithmetic Decoding (CPU)")
            
            # decoded_pixel_values = np.zeros((batch_size, k), dtype=np.int64)
            
            # # 串行解码 (Batch内循环)
            # for b in range(batch_size):
            #     dec = decoders[b]
            #     p_b = target_probs_cpu[b] # [K, 256]
            #     for i in range(k):
            #         try:
            #             pixel_val = dec.decode(normalize_pdf_for_arithmetic_coding(p_b[i]))
            #             if not (0<=pixel_val<256):
            #                 print(f"警告: 解码出错，像素值 {pixel_val} 越界，强制设为128")
            #                 pixel_val = 128  # 沿用D3PM设置，解码失败则为灰像素
            #         except Exception as e:
            #             print(f"错误: Batch {b} 第 {i} 个像素解码失败，设为128，错误信息: {e}")
            #             pixel_val = 128  # 沿用D3PM设置，解码失败则为灰像素
            #         decoded_pixel_values[b, i] = pixel_val
            
            # profiler.end_tick()
            # profiler.tick("Model Inference (GPU)")
            
            # # 7. 更新状态 (GPU)
            # # 将解码出的像素值转回 Token ID
            # decoded_pixels_tensor = torch.tensor(decoded_pixel_values, device=device, dtype=torch.long)
            # decoded_token_ids = ctx.pixel_to_token_tensor[decoded_pixels_tensor] # [B, K]
            
            # # 填入 xt
            # xt.scatter_(1, target_indices, decoded_token_ids)
            
            # # 更新 Mask (将已解码位置设为 False)
            # false_tensor = torch.zeros_like(decoded_token_ids, dtype=torch.bool)
            # maskable_mask.scatter_(1, target_indices, false_tensor)
            # # --------------------------------------------------------------------------------

    profiler.tick("Finalize")
    
    # 提取图像部分 (去除 BOS)
    # xt: [B, Seq_Len] -> pixel tokens
    output_tokens = xt[:, 1:] # [B, H*W*C]
    
    # 将 Token ID 转回 像素值    
    recovered_pixels = ctx.id_to_pixel_tensor[output_tokens] # [B, H*W*C]
    recovered_images_np = recovered_pixels.cpu().numpy().astype(np.uint8)
    
    # Reshape images
    reconstructed_batch = []
    for i in range(batch_size):
        img_flat = recovered_images_np[i]
        if constants.IS_CHANNEL_WISED:
            img = img_flat.reshape(h, w, 1)
        else:
            img = img_flat.reshape(h, w, 3)
        reconstructed_batch.append(img)
        
    profiler.end_tick()
    return reconstructed_batch


def reconstruct_image_from_patches(patches, image_shape):
    """
    将 patch 列表重组为完整图像。
    支持 Channel-wise (分通道) 和 Pixel-wise (全通道) 两种模式。
    """
    h_img, w_img, c_img = image_shape
    h_p, w_p = constants.CHUNK_SHAPE_2D
    
    # 计算网格布局
    n_rows = h_img // h_p
    n_cols = w_img // w_p
    patches_per_plane = n_rows * n_cols
    
    full_image = np.zeros((h_img, w_img, c_img), dtype=np.uint8)
    
    # 检查 Patch 的通道数
    patch_c = patches[0].shape[-1] if patches[0].ndim == 3 else 1
    
    if constants.IS_CHANNEL_WISED and c_img == 3 and patch_c == 1:
        # 分通道模式: 列表顺序为 [所有R Patch, 所有G Patch, 所有B Patch]
        for c in range(3):
            # 获取当前通道的所有 patch
            start_idx = c * patches_per_plane
            plane_patches = patches[start_idx : start_idx + patches_per_plane]
            
            idx = 0
            for r in range(n_rows):
                for col in range(n_cols):
                    if idx < len(plane_patches):
                        p = plane_patches[idx] # (h_p, w_p, 1)
                        if p.ndim == 3: p = p[:, :, 0]
                        full_image[r*h_p:(r+1)*h_p, col*w_p:(col+1)*w_p, c] = p
                        idx += 1
    else:
        # 标准模式 (或单通道图): Patch 顺序即为 Raster Scan 顺序
        idx = 0
        for r in range(n_rows):
            for col in range(n_cols):
                if idx < len(patches):
                    full_image[r*h_p:(r+1)*h_p, col*w_p:(col+1)*w_p, :] = patches[idx]
                    idx += 1
                    
    return full_image


# ==========================================
# 5. 主程序
# ==========================================
def main(args):
    print(f"模型路径: {args.model_path}")
    print(f"基础模型: {args.base_model_name}")
    
    tokenizer, model = load_ddm(args)
    ctx = CompressionContext(tokenizer)
    
    if not os.path.exists(args.input_dir):
        print(f"错误: 文件 {args.input_dir} 不存在，请先运行压缩脚本。")
        return
    
    # channel = AWGN/Rayleigh
    for SNR in range(12, -4, -3):
        print(f"\n{'='*30}\n开始解压 SNR={SNR} 的文件...")
        args.input_file = args.input_dir + '/demo_decode_SNR_' + str(SNR) + '.txt'
        save_dir_reconstructed = os.path.join(args.output_dir, f'SNR_{SNR}')
    
    # # channel = None
    # for SNR in [0]:
    #     args.input_file = args.input_dir + '/compressed_output.txt'
    #     save_dir_reconstructed = os.path.join(args.output_dir, 'recon_clean')
        
        os.makedirs(save_dir_reconstructed, exist_ok=True)
        print(f"正在读取压缩文件: {args.input_file}")
        with open(args.input_file, 'r') as f:
            lines = f.readlines()
    
        print(f"总 Patch 数: {len(lines)}")
        print(f"Batch Size: {constants.BATCH_SIZE}")
        print(f"🚀 开始高性能扩散解码 (True Batching + Stable Sort)...")
        
        start_time_total = time.time()
        
        # 简单的 Batch 迭代器
        batch_buffer = []
        current_patches = []
        image_idx = 0
        patches_per_image = constants.PATCHES_PER_IMAGE_TEST
        
        for i, line in enumerate(tqdm(lines)):
            raw_bit_string = line.strip()
            if not raw_bit_string:
                continue
            
            ###移除停止位
            # 1. 去除末尾所有的 '0' (Padding)
            stripped_padding = raw_bit_string.rstrip('0')
            
            # 2. 去除末尾的停止位 '1'
            if len(stripped_padding) > 0:
                bit_string = stripped_padding[:-1]
            else:
                print(f"警告: 第 {i} 行数据异常，未找到停止位")
                bit_string = ""
            batch_buffer.append(bit_string)
            
            if len(batch_buffer) == constants.BATCH_SIZE:
                recon_imgs = decompress_image_batched(batch_buffer, model, tokenizer, ctx, args)
                current_patches.extend(recon_imgs)
                batch_buffer = []
                
                while len(current_patches) >= patches_per_image:
                    img_patches = current_patches[:patches_per_image]
                    current_patches = current_patches[patches_per_image:]
                    
                    full_image = reconstruct_image_from_patches(img_patches, constants.IMAGE_SHAPE_TEST)
                    
                    img_save_path = os.path.join(save_dir_reconstructed, f'image_{image_idx}.png')
                    if full_image.shape[2] == 1: # Grayscale
                        Image.fromarray(full_image[:, :, 0], mode='L').save(img_save_path)
                    else:
                        Image.fromarray(full_image, mode='RGB').save(img_save_path)
                    print(f"保存图像: {img_save_path}")
                    image_idx += 1
                
                if i % (constants.BATCH_SIZE * 5) == 0 and args.verbose:
                    profiler.print_stats()
                    
        # 处理剩余的
        if len(batch_buffer) > 0:
            recon_imgs = decompress_image_batched(batch_buffer, model, tokenizer, ctx, args)
            current_patches.extend(recon_imgs)
            # 处理剩余的图像重组
            while len(current_patches) >= patches_per_image:
                img_patches = current_patches[:patches_per_image]
                current_patches = current_patches[patches_per_image:]
                
                full_image = reconstruct_image_from_patches(img_patches, constants.IMAGE_SHAPE_TEST)
                
                img_save_path = os.path.join(save_dir_reconstructed, f'image_{image_idx}.png')
                if full_image.shape[2] == 1:
                    Image.fromarray(full_image[:, :, 0], mode='L').save(img_save_path)
                else:
                    Image.fromarray(full_image, mode='RGB').save(img_save_path)
                print(f"保存图像: {img_save_path}")
                image_idx += 1
            
        end_time_total = time.time()
        
        profiler.print_stats()
        print(f"总耗时: {end_time_total - start_time_total:.2f}s")
        print(f"平均速度: {(end_time_total - start_time_total)/len(lines):.4f} s/patch")
        print("解码完成。")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffugpt-s', choices=['diffugpt-s', 'diffugpt-m', 'diffullama'])
    parser.add_argument("--base_model_name", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'llama'])
    parser.add_argument("--model_path", type=str, default='../Model', help="DiffuGPT path")
    parser.add_argument("--ddm_sft", type=bool, default=True, help="是否使用微调后的DiffuGPT模型")
    parser.add_argument("--checkpoint_dir", type=str, default='train_20251228_192149')
    parser.add_argument("--checkpoint_name", type=str, default='checkpoint-48000')
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--confidence_st", type=str, default='entropy', choices=['entropy', 'topk', 'simple'])
    parser.add_argument('--verbose', type=bool, default=False, help='打印详细过程')
    parser.add_argument("--channel", type=str, default='AWGN', choices=[None, 'AWGN', 'Rayleigh'])
    parser.add_argument("--dataset_type", type=str, default="DIV2K_LR_X4")
    parser.add_argument("--root_dir", type=str, default="./image_io", help="输入输出文件夹的根目录")
    
    args = parser.parse_args()
    
    args.model_path = os.path.join(args.model_path, args.model_name)
    if args.ddm_sft:
        args.model_path = os.path.join(args.model_path, "ddm-sft", args.checkpoint_dir, args.checkpoint_name)
    args.root_dir = os.path.join(args.root_dir, f'{args.dataset_type}', f'{args.confidence_st}_confidence')
    args.root_dir = os.path.join(args.root_dir, 'channel_indep') if constants.IS_CHANNEL_WISED else os.path.join(args.root_dir, 'channel_corre')
    if args.ddm_sft:
        args.root_dir = os.path.join(args.root_dir, f'patch{constants.CHUNK_SHAPE_2D}', f'{args.model_name}_ddm-sft', f'{args.checkpoint_dir}', f'diffu_step{args.diffusion_steps}')
    else:
        args.root_dir = os.path.join(args.root_dir, f'patch{constants.CHUNK_SHAPE_2D}', f'{args.model_name}', f'diffu_step{args.diffusion_steps}')
    
    if args.channel:
        args.input_dir  = os.path.join(args.root_dir, f"ECCT_forward/LDPC_K24_N49/{args.channel}")
        args.output_dir = os.path.join(args.root_dir, f"ECCT_reconstruct/LDPC_K24_N49/{args.channel}")
    else:
        args.input_dir  = args.root_dir
        args.output_dir = args.root_dir
    
    return args

if __name__ == "__main__":
    set_seed(42)
    main(get_args())