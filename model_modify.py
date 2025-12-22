import torch
from transformers import ImageGPTForCausalImageModeling

# 加载原始模型
model = ImageGPTForCausalImageModeling.from_pretrained("./data/model/igpt/igpt-l")

# 检查实际维度
print(f"原始wte形状: {model.transformer.wte.weight.shape}")
print(f"原始lm_head形状: {model.lm_head.weight.shape}")

# 修改配置
model.config.vocab_size = 257

# 获取实际的embedding维度
embedding_dim = model.transformer.wte.weight.shape[1]

# 创建新的embedding层
old_wte = model.transformer.wte
new_wte = torch.nn.Embedding(257, embedding_dim)

# 复制前257个token的权重
with torch.no_grad():
    min_dim = min(old_wte.weight.size(0), 257)
    new_wte.weight[:min_dim] = old_wte.weight[:min_dim]

model.transformer.wte = new_wte

# 修改lm_head层
old_lm_head = model.lm_head
new_lm_head = torch.nn.Linear(embedding_dim, 256, bias=False)

# 复制前256个输出的权重
with torch.no_grad():
    min_dim = min(old_lm_head.weight.size(0), 256)
    new_lm_head.weight[:min_dim] = old_lm_head.weight[:min_dim]

model.lm_head = new_lm_head

# 验证修改
print(f"新wte形状: {model.transformer.wte.weight.shape}")
print(f"新lm_head形状: {model.lm_head.weight.shape}")

# 保存修改后的模型
model.save_pretrained("./data/model/igpt/igpt-l-modified")