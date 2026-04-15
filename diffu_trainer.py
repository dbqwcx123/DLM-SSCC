import torch
import torch.nn.functional as F
import os
from utils.pixel_token_dict import *
from diffu_model import shift_logits, get_confidence_entropy
from llamafactory.train.ddm.trainer import CustomDiffusionTrainer, get_anneal_attn_mask

class ImageDiscreteDiffusionTrainer(CustomDiffusionTrainer):
    """
    专门针对图像压缩任务定制的 Diffusion Trainer。
    核心修改：
    1. 在计算 Loss 时，只关注 0-255 对应的 Token Logits，并在此范围内做归一化。
    2. 引入了基于难度的局部加权（类似 Focal Loss），迫使模型更关注高信息量、难预测的像素。
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

        self._transition_model = None
        self._transition_attention_mask = None
        
        # ==================== 难度加权因子 (Gamma) ====================
        # gamma > 0: 值越大，模型越将注意力集中在“预测概率低”的困难像素上
        # self.focal_gamma = 2.0 
        # =============================================================


    def transition(self, x_0, sigma, maskable_mask):
        """
        基于全可见 x_0 的内在信息密度（自回归熵）进行引导的掩码策略。
        低熵位置优先被 mask，高熵位置优先保留可见。
        """
        device = x_0.device
        batch_size, seq_len = x_0.shape
        move_chance = sigma
        
        # 当前应 mask 的数量
        num_maskable = maskable_mask.sum(dim=1)  # [B]
        num_to_mask = torch.round(move_chance * num_maskable.float()).long()
        num_to_mask = torch.clamp(num_to_mask, min=0)
        num_to_mask = torch.minimum(num_to_mask, num_maskable)

        was_training = self._transition_model.training
        self._transition_model.eval()
        
        with torch.no_grad():
            # 1) 直接在全可见 x_0 上前向
            raw_logits = self._transition_model(x_0, attention_mask=self._transition_attention_mask)
            logits_shifted = shift_logits(raw_logits)
            logits_pixel = logits_shifted[:, :, self.pixel_token_ids] 
            
        # 熵从小到大排序：低熵位置优先被 mask
        entropy = get_confidence_entropy(logits_pixel, None)
        entropy = entropy.masked_fill(~maskable_mask, float("inf"))
        sorted_indices = torch.argsort(entropy, dim=-1, descending=False, stable=True)
            
        move_indices = torch.zeros_like(maskable_mask, dtype=torch.bool)
        max_k = int(num_to_mask.max().item())

        if max_k > 0:
            target_indices = sorted_indices[:, :max_k]   # [B, K]

            arange_k = torch.arange(max_k, device=device).unsqueeze(0).expand(batch_size, -1)
            valid_mask = arange_k < num_to_mask.unsqueeze(1)   # [B, K]

            flat_indices = target_indices.reshape(-1)
            flat_valid = valid_mask.reshape(-1)

            batch_offsets = torch.arange(batch_size, device=device) * seq_len
            batch_offsets = batch_offsets.unsqueeze(1).expand(-1, max_k).reshape(-1)
            final_flat_indices = batch_offsets + flat_indices

            move_indices.view(-1)[final_flat_indices[flat_valid]] = True

        x_t = torch.where(move_indices, self.tokenizer.mask_token_id, x_0)
        
        if was_training:
            self._transition_model.train()
            
        return x_t
        
    
    def inner_forward(self, model, inputs, eval=False):
        """
        重写 inner_forward 以实现 Image-Specific 的 Loss 计算
        """
        x = inputs["input_ids"]
        # src_mask 用于标记“不应该被 Mask 的区域”（如条件部分），但在纯图像压缩中通常是全 False (全部可压缩)
        if "src_mask" not in inputs:
            src_mask = torch.zeros_like(x, dtype=torch.bool)
            src_mask[:, 0] = True  # 第一个 token 是 BOS，不被 Mask
        else:
            src_mask = inputs["src_mask"].bool()
            
        batch_size, seq_len = x.shape
        device = x.device

        # 确保映射表在正确的设备上
        if self.pixel_token_ids_tensor.device != device:
             self.pixel_token_ids_tensor = self.pixel_token_ids_tensor.to(device)
             self.vocab_map = self.vocab_map.to(device)

        num_timesteps = self.diff_args.diffusion_steps

        # --- 1. 采样时间步 t（连续时间训练） ---
        sampling_eps = 1e-3
        t = (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps
        sigma = self.noiser.total_noise(t)
        dsigma = self.noiser.rate_noise(t)

        # --- 2. 构造 attention mask ---
        if hasattr(model, "module"):  # Handle DDP
            get_embeds = model.module.get_embeds
        else:
            get_embeds = model.get_embeds

        attn_mask_ratio = 1.0  # 使用双向全注意力
        x_embed = get_embeds(x)
        attention_mask = get_anneal_attn_mask(
            seq_len,
            batch_size,
            dtype=x_embed.dtype,
            device=device,
            attn_mask_ratio=attn_mask_ratio
        )

        # --- 3. 把当前 batch 的 model / attention_mask 临时缓存给 transition ---
        self._transition_model = model
        self._transition_attention_mask = attention_mask

        # --- 4. 用 transition 构造 x_t ---
        x_t = self.transition(x, sigma[:, None], maskable_mask=~src_mask)

        # 用完清空
        self._transition_model = None
        self._transition_attention_mask = None

        # --- 5. 正常训练前向 ---
        raw_logits = model(x_t, attention_mask=attention_mask)

        # --- 6. 核心修改：Logit Restriction & Loss Calculation ---
        
        # (1) 只有被 Mask 掉的位置才计算 Loss
        loss_mask = (x_t == self.tokenizer.mask_token_id)
        
        # (2) 只提取 0-255 对应的列 [B, L, 256]
        if self.finetuning_args.shift:
            logits_shifted = shift_logits(raw_logits)
            logits_pixel = logits_shifted[:,:, self.pixel_token_ids]
        else:
            logits_pixel = raw_logits[:,:, self.pixel_token_ids]
        
        # (3) 准备 Targets
        target_pixel_indices = self.vocab_map[x] # [B, L]
        
        valid_target_mask = target_pixel_indices != -1
        loss_mask = loss_mask & valid_target_mask

        # (4) 计算 CrossEntropy
        pixel_logits_flat = logits_pixel[loss_mask] 
        targets_flat = target_pixel_indices[loss_mask] 

        if targets_flat.shape[0] == 0:
            final_loss = torch.tensor(0.0, device=device, requires_grad=True)
            unweighted_loss = torch.tensor(0.0, device=device)
        else:
            full_targets = target_pixel_indices.clone()
            full_targets[~loss_mask] = -1
            
            # 计算原始的 CrossEntropy (即负对数似然 NLL)
            # shape: [B, L]
            loss_per_token = F.cross_entropy(
                logits_pixel.view(-1, 256), 
                full_targets.view(-1), 
                reduction='none', 
                ignore_index=-1
            ).view(batch_size, seq_len)
            
            # 转为 float 确保精度
            loss_per_token = loss_per_token.float()
            dsigma = dsigma.float()

            # ====================================================================
            # # 基于预测难度的局部信息量加权 (Focal-like Weighting)
            # # 1. 还原模型对“真实类别”的预测概率 pt
            # # 对于 full_targets == -1 的位置，loss 为 0，对应的 pt = 1.0
            # pt = torch.exp(-loss_per_token)
            # # 2. 计算局部难度权重
            # # 如果模型猜得很准 (pt -> 1)，这是大面积背景/冗余信息，权重 -> 0
            # # 如果模型猜不准 (pt -> 0)，这是边缘/高频细节/高信息量像素，权重 -> 1
            # difficulty_weights = (1 - pt) ** self.focal_gamma
            # # 3. 将难度权重叠加到像素级的 NLL 上
            # loss_per_token_weighted = loss_per_token * difficulty_weights
            # ====================================================================

            # 7. 结合宏观的时间步扩散率 dsigma，计算最终的 batch 维度 Loss
            weighted_loss = (dsigma[:, None] * loss_per_token).sum() 
            num_active_tokens = loss_mask.sum().float()
            
            final_loss = weighted_loss / (num_active_tokens + 1e-5)
            
            # 记录原始未加权（纯信息论意义上）的 loss，方便监控真实压缩性能的变化
            unweighted_loss = loss_per_token.sum() / (num_active_tokens + 1e-5)

        # Logging
        if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
            if not eval and self.state.global_step % self.args.logging_steps == 0:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'custom/total_loss': final_loss.item(), 
                        'custom/unweighted_loss': unweighted_loss.item(),
                        # 额外监控难度权重的均值，看看模型是不是越学越自信
                        # 'custom/mean_difficulty_weight': difficulty_weights[loss_mask].mean().item() if 'difficulty_weights' in locals() and loss_mask.any() else 0.0
                    }, commit=False)

        return final_loss