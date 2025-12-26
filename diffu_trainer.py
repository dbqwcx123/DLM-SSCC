import torch
import torch.nn.functional as F
import os
from utils.pixel_token_dict import *
from diffu_model import shift_logits
from llamafactory.train.ddm.trainer import CustomDiffusionTrainer, get_anneal_attn_mask

class ImageDiscreteDiffusionTrainer(CustomDiffusionTrainer):
    """
    专门针对图像压缩任务定制的 Diffusion Trainer。
    核心修改：在计算 Loss 时，只关注 0-255 对应的 Token Logits，并在此范围内做归一化。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 预先计算 0-255 字符串对应的 Token IDs
        # 假设 tokenizer 在 self.tokenizer 中可用
        self.pixel_token_ids = compute_pixel_token_ids(self.tokenizer)
        self.token_id_to_pixel = compute_token_ids_to_pixel(self.tokenizer)
        self.pixel_token_ids_tensor = torch.tensor(self.pixel_token_ids).to(self.args.device)
        
        # 创建一个大的映射表用于快速查找: map[token_id] -> pixel_val (0-255)
        # 初始化为 -1 (Ignored)
        self.vocab_map = torch.full((len(self.tokenizer),), -1, dtype=torch.long).to(self.args.device)
        for tid, pixel in self.token_id_to_pixel.items():
            self.vocab_map[tid] = pixel

    def inner_forward(self, model, inputs, eval=False):
        """
        重写 inner_forward 以实现 Image-Specific 的 Loss 计算
        """
        x = inputs["input_ids"]
        # src_mask 用于标记“不应该被 Mask 的区域”（如条件部分），但在纯图像压缩中通常是全 False (全部可压缩)
        if "src_mask" not in inputs:
            src_mask = torch.zeros_like(x, dtype=torch.bool)
        else:
            src_mask = inputs["src_mask"].bool()
            
        batch_size, seq_len = x.shape
        device = x.device

        # 确保映射表在正确的设备上
        if self.pixel_token_ids_tensor.device != device:
             self.pixel_token_ids_tensor = self.pixel_token_ids_tensor.to(device)
             self.vocab_map = self.vocab_map.to(device)

        num_timesteps = self.diff_args.diffusion_steps
        
        # --- 1. 采样时间步 t ---
        sampling_eps = 1e-3
        t = (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps
        sigma = self.noiser.total_noise(t)
        dsigma = self.noiser.rate_noise(t)

        # --- 2. 扩散过程 (加噪/Masking) ---
        # 这一步与原逻辑相同：随机将部分 Token 替换为 [MASK]
        x_t = self.transition(x, sigma[:, None], maskable_mask=~src_mask)

        # --- 3. 模型前向传播 ---
        # 获取 Embeddings
        if hasattr(model, "module"): # Handle DDP
            get_embeds = model.module.get_embeds
        else:
            get_embeds = model.get_embeds
            
        # Attention Mask 设置 (Full Attention for Image)
        attn_mask_ratio = 1.0 # 使用双向全注意力
        x_embed = get_embeds(x) # 原始输入的 Embedding，用于长度等参考，实际 model forward 会处理 x_t
        attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=device, attn_mask_ratio=attn_mask_ratio)

        # 获取全词表 Logits [B, L, Vocab_Size]
        raw_logits = model(x_t, attention_mask=attention_mask)

        # --- 4. 核心修改：Logit Restriction & Loss Calculation ---
        
        # (1) 只有被 Mask 掉的位置才计算 Loss
        loss_mask = (x_t == self.tokenizer.mask_token_id)
        
        # (2) 只提取 0-255 对应的列 [B, L, 256]
        # pixel_logits = logits.index_select(dim=-1, index=self.pixel_token_ids_tensor)
        # 经过移位操作后，logits_shifted[:, i] 才是针对 xt[:, i] 的预测分布
        if self.finetuning_args.shift:
            logits_shifted = shift_logits(raw_logits)
            logits_pixel = logits_shifted[:,:, self.pixel_token_ids]
        else:
            logits_pixel = raw_logits[:,:, self.pixel_token_ids]
        
        # (3) 准备 Targets
        # 原始 x 包含的是 Token IDs (如 12345)，我们需要它在 0-255 中的索引 (如 128)
        # 使用预先构建的 vocab_map 进行转换
        target_pixel_indices = self.vocab_map[x] # [B, L]
        
        # 过滤掉非像素 Token (比如 BOS, EOS)，虽然数据加载时应该保证只有像素
        # 如果 target_pixel_indices 为 -1，则忽略
        valid_target_mask = target_pixel_indices != -1
        loss_mask = loss_mask & valid_target_mask

        # (4) 计算 CrossEntropy
        # Flatten for loss
        pixel_logits_flat = logits_pixel[loss_mask] # [N_masked, 256]
        targets_flat = target_pixel_indices[loss_mask] # [N_masked] values in 0-255

        if targets_flat.shape[0] == 0:
            final_loss = torch.tensor(0.0, device=device, requires_grad=True)
            unweighted_loss = torch.tensor(0.0, device=device)
        else:
            # 计算 Loss (自带 Softmax)
            # reduction='none' 以便我们可以应用 dsigma 权重
            # 但为了简化且符合 standard implementation，我们先算 mean 再乘权重，或者按 batch 算
            # 原代码是按 batch 维度加权，这里我们简化为全局，或者尽量贴合原代码逻辑
            
            # 原代码逻辑复刻：
            # loss = F.cross_entropy(..., reduction='none').reshape(batch_size, -1)
            # 这里由于我们 mask 比较复杂，直接算出来可能更好。
            # 为了保持 dsigma (时间步权重) 的作用，我们需要保留 batch 维度信息。
            
            # 重新组织计算方式：
            # 全量计算 CrossEntropy，利用 ignore_index 处理非 mask 区域
            # Create a target tensor with -1 everywhere except masked positions
            full_targets = target_pixel_indices.clone()
            full_targets[~loss_mask] = -1
            
            # [B, L]
            loss_per_token = F.cross_entropy(
                logits_pixel.view(-1, 256), 
                full_targets.view(-1), 
                reduction='none', 
                ignore_index=-1
            ).view(batch_size, seq_len)
            
            # 此时 loss_per_token 在非 mask 位置是 0
            
            # 加权求和
            # dsigma: [B] -> [B, 1]
            weighted_loss = (dsigma[:, None] * loss_per_token).sum() 
            num_active_tokens = loss_mask.sum()
            
            final_loss = weighted_loss / (num_active_tokens + 1e-6)
            unweighted_loss = loss_per_token.sum() / (num_active_tokens + 1e-6)

        # Logging
        if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
            if not eval and self.state.global_step % self.args.logging_steps == 0:
                import wandb
                if wandb.run is not None:
                    # 使用 commit=False 避免和 Trainer 的默认 log 产生时间戳冲突
                    wandb.log({
                        'custom/total_loss': final_loss.item(), 
                        'custom/unweighted_loss': unweighted_loss.item()
                    }, commit=False)

        return final_loss
    