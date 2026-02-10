import os
import torch
import torch.distributions as dists
from utils.attention_patch import replace_attention_mask

replace_attention_mask()

import transformers

from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
import torch.nn.functional as F
import torch.nn as nn
## add attention_patch
from huggingface_hub import PyTorchModelHubMixin

class DiscreteDiffusionModel(nn.Module, PyTorchModelHubMixin):
    """
    diffusion model
    """
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def __init__(
        self,
        model,
        config,
        tokenizer,
        device
    ):
        super().__init__()
        if isinstance(model, str): # if use pre-trained model name from huggingface, e.g., gpt2, gpt2-medium.
            config_pt = AutoConfig.from_pretrained(f'../Model/{model}')
            with torch.device(device):
                self.model = AutoModelForCausalLM.from_config(config_pt)
        else:
            self.model = model
        self.config = config
        self.embed_dim = self.config.hidden_size
        self.hidden_dim = self.config.hidden_size
        if self.model.get_input_embeddings().weight.size(0) != len(tokenizer):
            self.model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)
        self.vocab_size = self.model.get_input_embeddings().weight.size(0)
        if getattr(self.config, "model_type", None) == "gpt2":
            self.embed_tokens = self.model.transformer.wte  # 词嵌入层
            # use inputs_embeds instead of input_ids in forward function
            self.denoise_model = self.model.transformer  # 去噪主干网络
            for gpt2block in self.model.transformer.h:
                gpt2block.attn.bias.fill_(True)  # 移除因果掩码
            self.lm_head = self.model.lm_head  # 输出头
            del self.denoise_model.wte  # 删除原有的词嵌入层
        elif getattr(self.config, "model_type", None) == "llama":
            self.embed_tokens = self.model.model.embed_tokens
            self.denoise_model = self.model.model
            self.lm_head = self.model.lm_head
            del self.denoise_model.embed_tokens
        del self.model
        self.device = device

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def get_embeds(self, input_ids):
        return self.embed_tokens(input_ids)
    
    def forward(self, input_ids, attention_mask, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        denoise the input
        """
        # 获取输入嵌入 [batch_size, seq_len, hidden_dim]
        x_embed = self.get_embeds(input_ids)
        # 使用双向注意力进行去噪
        x = self.denoise_model(inputs_embeds = x_embed, attention_mask=attention_mask, return_dict = False)[0]
        # 输出logits [batch_size, seq_len, vocab_size]
        logits = self.get_logits(x)

        return logits


def get_anneal_attn_mask(seq_len, bsz, dtype, device, attn_mask_ratio):
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)
    
    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)

    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def get_confidence_entropy(logits, mask_indices):
    """
    计算掩码位置的置信度。
    
    原理：
    熵 (Entropy) 衡量了预测分布的不确定性。
    熵越低，分布越尖锐，不确定性越小，置信度应该越高。
    因此，我们返回 -Entropy 作为置信度。
    
    参数:
    logits: [batch, seq_len, vocab_size] 已经 shift 过的 logits
    mask_indices: [num_masks] 待解码的位置索引
    """
    # 为了数值稳定性，先计算 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    
    # 计算熵: H(x) = - sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=-1) # [batch, seq_len]
    selected_entropy = entropy[0, mask_indices]
    
    # 返回负熵 (熵越小，置信度越大)
    confidences = -selected_entropy
    # confidences = -entropy
    
    return confidences

def get_confidence_topk(logits, mask_indices):
    """
    计算掩码位置的置信度 (改进版：Top-K 概率总和)。
    反映了“真值落在前K个预测值中”的概率。
    """
    k = 5
    probs = torch.softmax(logits, dim=-1)
    
    topk_probs, _ = torch.topk(probs, k=k, dim=-1) # [batch, seq_len, k]
    mass_sum = torch.sum(topk_probs, dim=-1) # [batch, seq_len]
    
    # confidences = mass_sum[0, mask_indices]
    confidences = mass_sum
    
    return confidences

def get_confidence_simple(logits, mask_indices):
    """
    计算掩码位置的置信度。
    使用预测分布的最大概率值作为置信度。
    logits 必须已经经过 shift 处理，使得 logits[:, i] 对应 token[i] 的预测。
    """
    # logits: [batch, seq_len, vocab_size]
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1) # [batch, seq_len]
    # confidences = max_probs[0, mask_indices]
    confidences = max_probs
    return confidences


def conf_based_sorting(conf, mask_indices, device):
    """
    基于置信度对掩码位置进行排序。
    """
    # 将 Tensor 转为 CPU 上的列表，以免受 GPU 浮点不确定性影响
    conf_list = conf.detach().cpu().double().numpy().tolist()
    mask_idx_list = mask_indices.detach().cpu().numpy().tolist()
    
    # 构建 (confidence, original_index) 的元组列表
    combined = list(zip(conf_list, mask_idx_list))
    
    # confidence 降序 -> idx 升序 (Tie-breaker)
    combined.sort(key=lambda x: (-x[0], x[1]))
    
    # 提取排序后的 mask 位置
    sorted_mask_pos_list = [item[1] for item in combined]
    sorted_mask_pos = torch.tensor(sorted_mask_pos_list, device=device)
    
    return sorted_mask_pos


def batch_stable_sort(confidences_tensor):
    """
    替代 torch.sort。
    将 Batch 的置信度移至 CPU，转为 float64，并执行严格的稳定排序。
    Args:
        confidences_tensor: [Batch, Seq_Len] (GPU Tensor)
    Returns:
        sorted_indices: [Batch, Seq_Len] (GPU Tensor)
    """
    batch_size, seq_len = confidences_tensor.shape
    device = confidences_tensor.device
    
    # 1. 转移到 CPU 并转为 float64 以获得最大精度
    # numpy 处理 float64 比 tensor 转 list 更快一点
    conf_cpu = confidences_tensor.detach().cpu().double().numpy()
    
    sorted_indices_list = []
    
    # 2. 逐样本进行 Python 稳定排序
    for b in range(batch_size):
        row_conf = conf_cpu[b]
        # 构建 (confidence, index) 对
        combined = []
        for i in range(seq_len):
            combined.append((row_conf[i], i))
            
        # 排序规则：
        # 1. Confidence 降序 (-x[0])
        # 2. Index 升序 (x[1]) -> 这就是 Tie-breaker
        combined.sort(key=lambda x: (-x[0], x[1]))
        
        # 提取排序后的索引
        indices = [item[1] for item in combined]
        sorted_indices_list.append(indices)
        
    # 3. 转回 GPU
    return torch.tensor(sorted_indices_list, dtype=torch.long, device=device)


def shift_logits(logits):
    """
    DiffuLLaMA/DiffuGPT 的核心移位操作。
    原模型预测的是下一个 token，所以 logits[:, i] 实际上预测的是 input_ids[:, i+1]。
    我们需要将其向右移动一位，使得 logits_new[:, i+1] 来自 logits_old[:, i]。
    对于第0位，由于没有前文预测它，我们填入全0/忽略。
    """
    # logits shape: [batch, seq_len, vocab_size]
    # shift: [:, :-1] -> [:, 1:]
    shifted = torch.zeros_like(logits)
    shifted[:, 1:, :] = logits[:, :-1, :]
    return shifted


def smooth_probs(probs, k=1, alpha=0.2):
    """
    对概率分布进行局部平滑 (Label Smoothing / Convolution)。
    
    原理：
    如果模型预测 x 概率很高，我们认为 x 附近的 [x-k, x+k] 概率也应该适当提高。
    这样当真实值落在 x 旁边时，编码代价会显著降低。
    
    Args:
        probs: [..., 256] 原始概率分布 (Tensor)
        k: 邻域半径
        alpha: 平滑强度 (0~1)。 
               alpha越大，越多的概率质量会从峰值分散到邻居。
    """
    if k == 0:
        return probs
    
    # 1. 强制将输入数据移动到 CPU 并转为 float64 (双精度)
    probs = probs.detach().cpu().double()
    
    vocab_size = probs.shape[-1]
    original_shape = probs.shape
    
    # 原始分布转为 [Batch*Seq, 1, 256] 以便做 1D 卷积
    probs_flat = probs.view(-1, 1, vocab_size)
    
    # 2. 构造双精度卷积核：中心保留 (1-alpha)，两侧各分担 alpha/(2k)
    # 比如 k=1, alpha=0.2 -> [0.1, 0.8, 0.1]
    kernel_size = 2 * k + 1
    center_val = 1.0 - alpha
    side_val = alpha / (2 * k)
    
    # 显式指定 dtype=torch.float64，device='cpu'
    kernel = torch.ones(1, 1, kernel_size, dtype=torch.float64, device='cpu') * side_val
    kernel[0, 0, k] = center_val
    
    # 3. 双精度卷积
    # Padding 以保持维度一致 (Circular padding 或者是 Constant 0)
    # 像素是连续的数值，0和255不相邻，所以用 Constant Padding
    probs_padded = F.pad(probs_flat, (k, k), mode='constant', value=0.0)
    
    # 4. 执行平滑
    probs_smoothed = F.conv1d(probs_padded, kernel)
    
    # 5. 重新归一化 (因为 Padding 边缘可能导致概率和 < 1)
    probs_smoothed = probs_smoothed / (torch.sum(probs_smoothed, dim=-1, keepdim=True) + 1e-10)
    
    return probs_smoothed.view(original_shape)


def load_ddm(args):
    """
    加载离散扩散模型
    """
    model_name = args.model_path
    config = AutoConfig.from_pretrained(f'../Model/{args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    bin_file = os.path.join(model_name, "pytorch_model.bin")
    use_manual_loading = os.path.exists(bin_file) and not os.path.exists(os.path.join(model_name, "model.safetensors"))
    
    if use_manual_loading:
        print(f"检测到 pytorch_model.bin 且无 safetensors，使用手动模式加载模型权重")
    
        if args.base_model_name in ['gpt2', 'gpt2-medium']:
            with torch.device('cuda'):
                model = DiscreteDiffusionModel(
                    model=args.base_model_name, 
                    config=config, 
                    tokenizer=tokenizer,
                    device='cuda'
                )
        elif args.base_model_name == 'llama':
            # ... LLaMA 的手动加载逻辑 ...
            pass
        else:
            raise ValueError(f"未知的基础模型: {args.base_model_name}")
        state_dict = torch.load(bin_file, map_location='cuda')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"权重加载完成。丢失键: {len(missing)}, 多余键: {len(unexpected)}")
    else:
        print(f"使用默认 from_pretrained 加载模型")
        if args.base_model_name in ['gpt2', 'gpt2-medium']:
            with torch.device('cuda'):
                model = DiscreteDiffusionModel.from_pretrained(
                    model_name, 
                    model=args.base_model_name, 
                    config=config, 
                    tokenizer=tokenizer,
                    device='cuda'
                )
        elif args.base_model_name == 'llama':
            torch_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"  # linux
            
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
                _attn_implementation=attn_implementation,
                torch_dtype=torch_dtype
            )
        
            model = DiscreteDiffusionModel(
                model=model, 
                config=config, 
                tokenizer=tokenizer,
                device='cuda'
            ).to('cuda')
    
    return tokenizer, model