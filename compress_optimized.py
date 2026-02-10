import os
# 1. 解决 Tokenizers 死锁警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from diffu_model import *
import constants, data_loaders
from utils import arithmetic_coder
from utils.ac_utils import normalize_pdf_for_arithmetic_coding
from utils.ECCT_utils import set_seed
from utils.pixel_token_dict import *

# ==========================================
# TimeProfiler 类
# ==========================================
class TimeProfiler:
    def __init__(self, enabled=True):
        self.records = {}
        self.enabled = enabled
        self.current_start = None
        self.current_name = None

    def tick(self, name):
        if not self.enabled: return
        torch.cuda.synchronize()  # 确保 GPU 同步
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
        print("⚡ 性能耗时分布报告 (Time Profiling)")
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
# 压缩上下文
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

# ==========================================
# 核心：真·Batch 压缩逻辑
# ==========================================
def compress_image(batch_pixel_arrays, model, tokenizer, ctx, args):
    """
    batch_pixel_arrays: List[np.ndarray], 每个元素是 flatten 的像素数组 (int)
    """
    profiler.tick("Setup & Tokenize (Vectorized)")
    model.eval()
    device = model.device
    batch_size = len(batch_pixel_arrays)

    # 1. 向量化输入构建
    pixels_batch = torch.tensor(np.array(batch_pixel_arrays), dtype=torch.long, device=device)
    input_ids_body = ctx.pixel_to_token_tensor[pixels_batch]
    
    bos_token = torch.tensor([tokenizer.bos_token_id], device=device).expand(batch_size, 1)
    x = torch.cat([bos_token, input_ids_body], dim=1) # [B, Seq_Len + 1]
    
    seq_len = x.size(1)
    
    src_mask = torch.zeros_like(x, dtype=torch.bool).to(device)
    maskable_mask = ~src_mask
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=model.denoise_model.dtype, device=device, attn_mask_ratio=1.0)
    
    maskable_mask[:, 0] = False
    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)
    
    encoders = []
    output_bits_lists = []
    for _ in range(batch_size):
        buf = []
        output_bits_lists.append(buf)
        encoders.append(arithmetic_coder.Encoder(
            base=constants.ARITHMETIC_CODER_BASE,
            precision=constants.ARITHMETIC_CODER_PRECISION,
            output_fn=buf.append,
        ))

    profiler.tick("Model Inference (GPU)") 

    with torch.inference_mode():
        for t in range(args.diffusion_steps-1, -1, -1):
            # 1. GPU 推理
            # with torch.cuda.amp.autocast(enabled=True):  # 混合精度
            with torch.no_grad():
                raw_logits = model(xt, attention_mask=attention_mask)
            
            # 2. Logits 处理
            logits_shifted = shift_logits(raw_logits)
            logits_pixel = logits_shifted[:,:, ctx.pixel_token_ids]
            
            # 3. 置信度计算
            if args.confidence_st == 'entropy':
                confidences = get_confidence_entropy(logits_pixel, None)
            else:
                probs = F.softmax(logits_pixel, dim=-1)
                max_probs, _ = probs.max(dim=-1)
                confidences = max_probs
            
            confidences = confidences.masked_fill(~maskable_mask, float('-inf'))
            
            # -----------------------------------------------------------------------------
            # 计算每个样本实际剩余的 mask 数量 [B]
            masks_left_per_sample = torch.sum(maskable_mask, dim=1) 
            num_current_masks = masks_left_per_sample.max().item()
            if num_current_masks == 0: break
            
            ratio = 1.0 / (t + 1)
            k = max(1, min(int(num_current_masks * ratio), num_current_masks))
            
            # 排序获取 indices [B, K]
            sorted_indices = batch_stable_sort(confidences)
            target_indices = sorted_indices[:, :k]
            
            # 4. 提取数据
            batch_indices_expanded = target_indices.unsqueeze(-1).expand(-1, -1, logits_pixel.size(-1))
            target_logits = torch.gather(logits_pixel, 1, batch_indices_expanded)
            target_probs = torch.softmax(target_logits.double(), dim=-1)
            true_token_ids = torch.gather(x, 1, target_indices)
            
            # 5. 更新输入 xt (仅 GPU 更新部分，先不急着 scatter)
            # 我们需要构建一个 mask，只更新那些实际上还有 mask 的位置
            # 如果某个样本只剩 2 个 mask，但在 k=10 的循环里，后 8 个 indices 是无效的
            
            # 构建有效性 Mask [B, K]
            # 这里的逻辑是：如果当前列索引 i < 该样本剩余 mask 数，则有效
            arange_k = torch.arange(k, device=device).unsqueeze(0).expand(batch_size, -1)
            masks_left_expanded = masks_left_per_sample.unsqueeze(1).expand(-1, k)
            valid_k_mask = arange_k < masks_left_expanded  # [B, K] bool

            # 仅在有效位置更新 xt
            # 使用 view(-1) 展平处理来避免复杂的 gather/scatter 逻辑
            flat_indices = target_indices.reshape(-1)
            flat_src = true_token_ids.reshape(-1)
            flat_mask = valid_k_mask.reshape(-1)
            
            # 只有 valid 的位置才进行 scatter 更新
            # xt.scatter_ 无法直接接受 mask，所以我们用 tensor 索引操作
            # 展平后的 xt 索引
            batch_offsets = torch.arange(batch_size, device=device) * xt.size(1)
            batch_offsets = batch_offsets.unsqueeze(1).expand(-1, k).reshape(-1)
            final_flat_indices = batch_offsets + flat_indices
            
            # 执行更新：只更新 valid_k_mask 为 True 的部分
            xt.view(-1)[final_flat_indices[flat_mask]] = flat_src[flat_mask]
            
            # 更新 maskable_mask (设为 False)
            maskable_mask.view(-1)[final_flat_indices[flat_mask]] = False
            
            # 6. 数据传回 CPU
            target_probs_cpu = target_probs.cpu().numpy()
            true_pixel_vals_gpu = torch.gather(ctx.id_to_pixel_tensor, 0, true_token_ids.view(-1)).view(batch_size, k)
            true_pixel_vals_cpu = true_pixel_vals_gpu.cpu().numpy()
            
            # 传递到 CPU 的剩余数量
            masks_left_cpu = masks_left_per_sample.cpu().numpy()

            profiler.end_tick()
            profiler.tick("Arithmetic Coding (CPU)")
            
            # 串行编码 (加入有效性判断)
            for b in range(batch_size):
                enc = encoders[b]
                p_b = target_probs_cpu[b]
                pix_b = true_pixel_vals_cpu[b]
                
                # 【关键修复】：取 k 和 实际剩余数量 的最小值
                actual_k = min(k, int(masks_left_cpu[b]))
                
                for i in range(actual_k):
                    enc.encode(normalize_pdf_for_arithmetic_coding(p_b[i]), int(pix_b[i]))
            
            profiler.end_tick()
            profiler.tick("Model Inference (GPU)")
            
            # # -----------------------------------------------------------------------------
            # num_current_masks = torch.sum(maskable_mask, dim=1).max().item()
            # if num_current_masks == 0: break
            
            # ratio = 1.0 / (t + 1)
            # k = max(1, min(int(num_current_masks * ratio), num_current_masks))
            
            # # 稳定排序策略
            # # 当 confidences 中有相同值时，stable=True 保证保留原始索引顺序(即优先选前面的)
            # # _, sorted_indices = torch.sort(confidences, descending=True, stable=True)
            # sorted_indices = batch_stable_sort(confidences)
            # target_indices = sorted_indices[:, :k]
            
            # # 4. 提取数据
            # batch_indices_expanded = target_indices.unsqueeze(-1).expand(-1, -1, logits_pixel.size(-1))
            # target_logits = torch.gather(logits_pixel, 1, batch_indices_expanded)
            # # target_logits = target_logits.detach().cpu().double()
            # target_probs = torch.softmax(target_logits.double(), dim=-1)
            
            # true_token_ids = torch.gather(x, 1, target_indices)
            
            # # 5. 更新输入 xt
            # xt.scatter_(1, target_indices, true_token_ids)
            # false_tensor = torch.zeros_like(true_token_ids, dtype=torch.bool)
            # maskable_mask.scatter_(1, target_indices, false_tensor)
            
            # # 6. 数据传回 CPU
            # target_probs_cpu = target_probs.cpu().numpy()
            # true_pixel_vals_gpu = torch.gather(ctx.id_to_pixel_tensor, 0, true_token_ids.view(-1)).view(batch_size, k)
            # true_pixel_vals_cpu = true_pixel_vals_gpu.cpu().numpy()
            
            # profiler.end_tick()
            # profiler.tick("Arithmetic Coding (CPU)")
            
            # # 串行编码
            # for b in range(batch_size):
            #     enc = encoders[b]
            #     p_b = target_probs_cpu[b]
            #     pix_b = true_pixel_vals_cpu[b]
            #     for i in range(k):
            #         enc.encode(normalize_pdf_for_arithmetic_coding(p_b[i]), int(pix_b[i]))
            
            # profiler.end_tick()
            # profiler.tick("Model Inference (GPU)")
            # # -----------------------------------------------------------------------------

    profiler.tick("Finalize")
    final_bits_list = []
    for enc, buf in zip(encoders, output_bits_lists):
        enc.terminate()
        compressed_bits = "".join(map(str, buf))
        final_bits_list.append(compressed_bits)
    
    profiler.end_tick()
    return final_bits_list, seq_len

# ==========================================
# 主程序
# ==========================================
def main(args):
    print(f"模型路径: {args.model_path}")
    print(f"基础模型: {args.base_model_name}")
    
    tokenizer, model = load_ddm(args)
    ctx = CompressionContext(tokenizer)
    
    print(f"图像数据集路径: {args.input_path}")
    print(f"处理图像块形状: {constants.CHUNK_SHAPE_2D}, Batch Size: {constants.BATCH_SIZE}")
    
    data_iterator = data_loaders.get_image_iterator(
                    num_chunks=constants.NUM_CHUNKS_TEST,
                    is_channel_wised=constants.IS_CHANNEL_WISED,
                    is_seq=False,
                    data_path=args.input_path)
    
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, 'compressed_output.txt')
    with open(output_file, "w") as f: pass
    
    idx = 0
    total_bits = 0
    h, w = constants.CHUNK_SHAPE_2D
    channels = 1 if constants.IS_CHANNEL_WISED else 3
    
    BATCH_SIZE = constants.BATCH_SIZE
    batch_buffer = [] 
    
    print(f"🚀 开始高性能扩散压缩 (True Batching + Stable Sort), T={args.diffusion_steps}...")
    start_time_total = time.time()
    
    try:
        for data, frame_id in tqdm(data_iterator):
            seq_array = data.reshape(1, h*w*channels)
            flattened_array = seq_array.flatten()
            batch_buffer.append(flattened_array)
            
            if len(batch_buffer) == BATCH_SIZE:
                compressed_bits_list, seq_len = compress_image(batch_buffer, model, tokenizer, ctx, args)
                
                with open(output_file, "a") as f:
                    for bits in compressed_bits_list:
                        bits_to_write = bits + '1'
                        f.write(bits_to_write + '\n')
                        total_bits += len(bits_to_write)
                
                idx += len(batch_buffer)
                batch_buffer = []
                
                if idx % (BATCH_SIZE * 5) == 0 and args.verbose:
                    profiler.print_stats()
            
        if len(batch_buffer) > 0:
            compressed_bits_list, seq_len = compress_image(batch_buffer, model, tokenizer, ctx, args)
            with open(output_file, "a") as f:
                for bits in compressed_bits_list:
                    bits_to_write = bits + '1'
                    f.write(bits_to_write + '\n')
                    total_bits += len(bits_to_write)
            idx += len(batch_buffer)

    except KeyboardInterrupt:
        print("\n用户中断，正在打印当前性能报告...")
    
    end_time_total = time.time()
    
    profiler.print_stats()
    
    print("\n" + "="*30)
    print(f"总处理 Patch 数: {idx}")
    print(f"总压缩比特数: {total_bits}")
    print(f"总耗时: {end_time_total - start_time_total:.2f}s")
    if idx > 0:
        print(f"平均速度: {(end_time_total - start_time_total)/idx:.4f} s/patch")
    print(f"压缩文件已保存至: {output_file}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffugpt-m', choices=['diffugpt-s', 'diffugpt-m', 'diffullama'])
    parser.add_argument("--base_model_name", type=str, default='gpt2-medium', choices=['gpt2', 'gpt2-medium', 'llama'])
    parser.add_argument("--model_path", type=str, default='../Model', help="DiffuGPT path")
    parser.add_argument("--ddm_sft", type=bool, default=True, help="是否使用微调后的DiffuGPT模型")
    parser.add_argument("--checkpoint_dir", type=str, default='train_full_20260117_005324')
    parser.add_argument("--checkpoint_name", type=str, default='checkpoint-56000')
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--confidence_st", type=str, default='entropy', choices=['entropy', 'topk', 'simple'])
    parser.add_argument('--verbose', type=bool, default=False, help='打印详细过程')
    parser.add_argument("--dataset_type", type=str, default="DIV2K_LR_X4")
    parser.add_argument("--input_path", type=str, default="../Dataset")
    parser.add_argument("--output_path", type=str, default="./image_io")
    args = parser.parse_args()
    
    args.model_path = os.path.join(args.model_path, args.model_name)
    if args.ddm_sft:
        args.model_path = os.path.join(args.model_path, "ddm-sft", args.checkpoint_dir, args.checkpoint_name)
    
    if args.dataset_type == "CIFAR10":
        args.input_path = os.path.join(args.input_path, "CIFAR10", "cifar10_test")
        # constants.IMAGE_SHAPE_TEST = (32, 32, 3)
    elif args.dataset_type == "DIV2K_HR":
        args.input_path = os.path.join(args.input_path, "DIV2K", "DIV2K_HR_test")
        # constants.NUM_IMAGE_TEST = 1
        # constants.IMAGE_SHAPE_TEST = (1024, 1024, 3)
    elif args.dataset_type == "DIV2K_LR_X2":
        args.input_path = os.path.join(args.input_path, "DIV2K", "DIV2K_LR_test/X2")
        # constants.IMAGE_SHAPE_TEST = (512, 512, 3)
    elif args.dataset_type == "DIV2K_LR_X4":
        args.input_path = os.path.join(args.input_path, "DIV2K", "DIV2K_LR_test/X4")
        # constants.IMAGE_SHAPE_TEST = (256, 256, 3)
    elif args.dataset_type == "DIV2K_LR_test":
        args.input_path = os.path.join(args.input_path, "DIV2K", "DIV2K_LR_test/test")
        # constants.NUM_IMAGE_TEST = 1
        # constants.IMAGE_SHAPE_TEST = (256, 256, 3)
    elif args.dataset_type == "ImageNet":
        args.input_path = os.path.join(args.input_path, "ImageNet", "test_unified")
        # constants.IMAGE_SHAPE_TEST = (256, 256, 3)
    
    args.output_path = os.path.join(args.output_path, f'{args.dataset_type}', f'{args.confidence_st}_confidence')
    args.output_path = os.path.join(args.output_path, 'channel_indep') if constants.IS_CHANNEL_WISED else os.path.join(args.output_path, 'channel_corre')
    if args.ddm_sft:
        args.output_path = os.path.join(args.output_path, f'patch{constants.CHUNK_SHAPE_2D}', f'{args.model_name}_ddm-sft', f'{args.checkpoint_dir}', f'diffu_step{args.diffusion_steps}')
    else:
        args.output_path = os.path.join(args.output_path, f'patch{constants.CHUNK_SHAPE_2D}', f'{args.model_name}', f'diffu_step{args.diffusion_steps}')
    return args

if __name__ == "__main__":
    set_seed(42)
    main(get_args())