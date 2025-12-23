import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ================= 配置区域 =================
MODEL_NAME = "NousResearch/Llama-2-7b-hf" # 或者你本地 Llama-7B 的路径

# 为了方便测试显存，这里直接生成伪造数据，
# 如果你想测试真实数据，请取消注释并替换为你的 txt/json 路径
USE_DUMMY_DATA = True
DATASET_PATH = "./your_data.txt" 
# ===========================================



def main():
    print(f"Loading Llama-7B in FP16 from {MODEL_NAME}...")
    
    llama_cache_dir = "E:/1Master1/0Science_Research/0DLM-SSCC/Model/Llama-2-7b-hf"
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=llama_cache_dir)
    tokenizer.pad_token = tokenizer.eos_token # Llama 没有 pad token，需要指定

    # 2. 以 FP16 加载模型 (显存占用大户)
    # device_map="auto" 会尝试将模型分配到 GPU，如果放不下会分流到 CPU/硬盘 (但这会导致训练极慢)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, 
        device_map="auto",
        cache_dir=llama_cache_dir
    )
    print(model.hf_device_map)
    print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # 3. 开启梯度检查点 (关键：大幅节省显存)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 4. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,            # LoRA 秩
        lora_alpha=128,  # LoRA 缩放因子
        lora_dropout=0.1
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # 打印可训练参数量

    # 5. 准备数据
    if USE_DUMMY_DATA:
        print("Constructing dummy dataset for VRAM testing...")
        data = [{"text": "Hello world, this is a test." * 20} for _ in range(100)]
        dataset = Dataset.from_list(data)
    else:
        from datasets import load_dataset
        dataset = load_dataset('text', data_files=DATASET_PATH)['train']

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=768)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir="./llama_lora_output",
        per_device_train_batch_size=1, # 显存捉襟见肘时设为 1
        gradient_accumulation_steps=4, # 通过累积模拟大 Batch
        learning_rate=2e-4,
        fp16=True, # 开启混合精度训练
        logging_steps=1,
        max_steps=10, # 只是为了测试显存，跑几步就行
        save_strategy="no",
        optim="adamw_torch" # Windows 下 bitsandbytes 的优化器可能兼容性不好，使用原生 torch 优化器
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training to test VRAM usage...")
    trainer.train()
    print("Training test finished successfully!")

if __name__ == "__main__":
    main()