from __future__ import print_function
import argparse
import random
import os
import numpy as np
from torch.utils import data
import logging
import torch
import time
from pyldpc import make_ldpc
from utils.ECCT_utils import *
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import constants

##################################################################

class My_Dataset(data.Dataset):
    def __init__(self, message_list, code, sigma, channel_type):
        """
        1. 接收所有消息列表，统一进行预处理。
        2. 预先计算编码(G矩阵乘法)，避免在GetItem中重复计算。
        """
        self.code = code
        self.sigma = sigma
        self.channel_type = channel_type
        
        # 预转换矩阵为 Float Tensor 以加速计算
        self.generator_matrix = code.generator_matrix.transpose(0, 1).float()
        self.pc_matrix = code.pc_matrix.transpose(0, 1).float()

        # 数据展平与预编码
        # 将所有消息合并处理，最大化 Batch 效率
        all_m = []
        for msg in message_list:
            groups = split_message(msg, self.code.k)
            all_m.extend(groups)
        
        # 转换为 Tensor (Total_Blocks, K)
        self.m_tensor = torch.tensor(all_m, dtype=torch.float)
        
        # 预先进行信道编码 x = mG (Total_Blocks, N)
        self.x_tensor = torch.matmul(self.m_tensor, self.generator_matrix) % 2

    def __getitem__(self, index):
        """
        只处理单个样本，Batch 组装交给 DataLoader
        """
        m = self.m_tensor[index]
        x = self.x_tensor[index]
        
        z = torch.randn(self.code.n) * self.sigma[0]
        
        if self.channel_type == 'AWGN':
            h = 1.0
        elif self.channel_type == 'Rayleigh':
            h = torch.from_numpy(np.random.rayleigh(1, self.code.n)).float()
        else:
            raise ValueError("Invalid channel type.")
        
        y = h * bin_to_sign(x) + z
        
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long().float(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        
        return m, x, z, y, magnitude, syndrome

    def __len__(self):
        return len(self.m_tensor)


##################################################################

def read_message_from_txt(filename):
    """读取消息文件并返回消息列表和统计信息"""
    message_list = []
    with open(filename, 'r') as f:
        message_string_list = f.read().splitlines()
        for message_string in message_string_list:
            if not message_string.strip(): continue # 跳过空行
            message = [int(x) for x in message_string]
            message_list.append(message)
    
    # 计算统计信息
    message_len = sum(len(msg) for msg in message_list)
    codeword_len = sum((len(msg) + code.k - 1) // code.k * code.n for msg in message_list)
    
    return message_list, message_len, codeword_len

def split_message(message, k):
    """将消息分割成k长度的组，不足的补零"""
    n = len(message)
    num_groups = (n + k - 1) // k
    result = []
    for i in range(num_groups):
        start = i * k
        end = min((i + 1) * k, n)
        group = message[start:end]
        if len(group) < k:
            group.extend([0] * (k - len(group)))
        result.append(group)
    return result  # num_groups=len(result)

def reconstruct_lines(all_pred_bits, message_list, code):
    """
    将扁平化的预测比特流恢复为原始的多行格式
    """
    reconstructed_lines = []
    bit_pointer = 0
    N, K = code.n, code.k
    
    # 遍历原始消息列表，利用其长度信息进行切分
    for original_msg in message_list:
        # 1. 计算当前这行消息原本占用了多少个 Block
        msg_len = len(original_msg)
        num_blocks = (msg_len + K - 1) // K
        
        # 2. 从长流中切出属于这一行的片段
        # chunk 是 N * num_blocks 长度的列表
        chunk = all_pred_bits[bit_pointer : bit_pointer + num_blocks * N]
        bit_pointer += num_blocks * N
        
        # 3. 解码：从每个 N 长的码字中提取前 K 个信息位
        current_line_str = ""
        for i in range(0, len(chunk), code.n):
            block = chunk[i : i + code.n]
            # 提取信息位 (假设是系统码，前K位是信息)
            info_bits = block[:code.k]
            current_line_str += "".join(map(str, info_bits))
            
        # 4. 去除末尾的填充0
        current_line_str = current_line_str.rstrip('0')            
        reconstructed_lines.append(current_line_str)
        
    return reconstructed_lines

##################################################################

def estimate(model, device, test_loader, code):
    """
    1. DataLoader 利用 Batch 进行推理。
    2. 使用 GPU 批量计算 Loss 和 Metric。
    """
    model.eval()
    
    total_loss = 0.
    total_ber = 0.
    total_fer = 0.
    total_samples = 0 # 统计实际的 Block 数量
    
    all_x_pred_bits = []  # 收集所有预测的比特
    
    with torch.no_grad():
        for (m, x, z, y, magnitude, syndrome) in tqdm(test_loader, desc="Inference", leave=False):
            m, x, z, y = m.to(device), x.to(device), z.to(device), y.to(device)
            magnitude, syndrome = magnitude.to(device), syndrome.to(device)
            
            # 前向传播 (Batch处理)
            z_pred = model(magnitude, syndrome)
            
            # 计算 Loss
            z_mul = (y * bin_to_sign(x))
            loss, x_pred = model.loss(z_pred, z_mul, y)

            # 统计指标
            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_ber += BER(x_pred, x) * batch_size
            total_fer += FER(x_pred, x) * batch_size
            total_samples += batch_size
            
            # 收集预测结果
            # x_pred shape: (Batch, N)
            x_pred_np = x_pred.cpu().numpy().astype(int)
            all_x_pred_bits.extend(x_pred_np.flatten().tolist())
            
    # 计算平均值
    avg_loss = total_loss / total_samples
    avg_ber = total_ber / total_samples
    avg_fer = total_fer / total_samples
    
    return avg_loss, avg_ber, avg_fer, all_x_pred_bits

##################################################################

class Code():
    pass

def main(args):
    # 路径设置和验证
    model_dir = f'../Model/Results_ECCT/{args.code_type}__Code_n_{args.code_n}_k_{args.code_k}_{args.channel}'
    msg_dir = f'./image_io/{mode}/diffu_step{diffu_step}/{args.msg_filename}.txt'
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model_dir {model_dir} does not exist.")
    if not os.path.exists(msg_dir):
        raise FileNotFoundError(f"Message_dir {msg_dir} does not exist.")
    
    args.model_path = model_dir
    args.msg_path = msg_dir
    
    print(f"Path to model\logs: {model_dir}")
    print(args)
    
    message_list, message_len, codeword_len = read_message_from_txt(args.msg_path)
    print(f'Message total length: {message_len}')
    print(f'Codeword total length: {codeword_len}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.load(os.path.join(args.model_path, 'best_model'), map_location='cpu')
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model.to(device)
    
    print(f'Transmission channel type: {args.channel}')
    
    SNR_range_test = np.arange(-1,10,2)
    ## TODO: 这里需要根据 baseline 调整 codeword_len
    _out_channel = 2*8
    _float_base = 32
    codeword_len_unified = _out_channel * (constants.IMAGE_SHAPE_TEST[0]//4) * (constants.IMAGE_SHAPE_TEST[1]//4) * _float_base * constants.NUM_IMAGE_TEST
    print(f'Codeword length unified: {codeword_len_unified}')
    SNR_range_test_real = SNR_range_test + 10*np.log10(codeword_len_unified/codeword_len)
    # SNR_range_test_real = SNR_range_test
    print(f'SNR_range_test: {SNR_range_test_real}')
    
    std_test = [SNR_to_std(ii) for ii in SNR_range_test_real]
    
    final_loss, final_ber, final_fer = [], [], []
    for ii, snr_val in tqdm(enumerate(SNR_range_test_real)):
        print(f"\n--- Testing SNR Index {ii}: {SNR_range_test[ii]} dB (Real: {snr_val:.2f} dB) ---")
        
        # 1. 创建 Dataset（包含所有消息）和 DataLoader
        test_dataset = My_Dataset(message_list, code, sigma=[std_test[ii]], channel_type=args.channel)
        test_loader = data.DataLoader(
            test_dataset, 
            batch_size=args.test_batch_size, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True
        )
        
        # 2. 执行推理 (获得扁平化的结果 all_pred_bits)
        loss, ber, fer, all_pred_bits = estimate(model, device, test_loader, code)
        
        final_loss.append(loss)
        final_ber.append(ber)
        final_fer.append(fer)
        
        print(f"Result -> BER: {ber:.4e} | FER: {fer:.4e} | Loss: {loss:.4f}")
        
        # 3. 保存解码后的信息流并恢复多行结构
        recovered_lines = reconstruct_lines(all_pred_bits, message_list, code)
        
        output_extracted_filename = f'./image_io/{mode}/diffu_step{diffu_step}/ECCT_forward/{code.code_name}/{channel}/demo_decode_SNR_{SNR_range_test[ii]}.txt'
        os.makedirs(os.path.dirname(output_extracted_filename), exist_ok=True)
        
        # 写入文件：现在 recovered_lines 的长度应该和 message_list 一样
        with open(output_extracted_filename, 'w') as f:
            for line in recovered_lines:
                f.write(line + "\n")
                
    metric_filename = f'./image_io/{mode}/diffu_step{diffu_step}/ECCT_forward/{code.code_name}/{channel}/#demo_decode_metric.txt'
    os.makedirs(os.path.dirname(metric_filename), exist_ok=True)
    with open(metric_filename, "w") as f:
        f.write("loss:\n"+ ", ".join([f"{x:.8e}" for x in final_loss]))
        f.write("\nBER:\n"+ ", ".join([f"{x:.8e}" for x in final_ber]))
        f.write("\nFER:\n"+ ", ".join([f"{x:.8e}" for x in final_fer]))
        
    print(f"\nTesting completed. Results saved to ./image_io/{mode}/...")

##################################################################
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ECCT')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default="0", help='gpus ids')
    parser.add_argument('--test_batch_size', type=int, default=1024)

    # Message args
    parser.add_argument('--msg_filename', type=str, default='compressed_output')

    # Code args
    parser.add_argument('--code_type', type=str, default='LDPC', choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=24)
    parser.add_argument('--code_n', type=int, default=49)
    parser.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--mode', type=str, default='DIV2K_LR_X4/entropy_confidence/channel_corre/patch(16, 16)/diffugpt-s_ddm-sft/train_20251228_192149')
    parser.add_argument('--diffu_step', type=int, default=20)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    # 环境设置
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(42)
    
    # 代码参数初始化
    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    
    # LDPC参数
    d_c = int(np.sqrt(code.n))
    d_v = int(d_c + 1 - code.k/(d_c-1))
    H, G = make_ldpc(code.n, d_v, d_c, systematic=True, sparse=True)
    code.generator_matrix = torch.from_numpy(G).long()
    code.pc_matrix = torch.from_numpy(H).long()
    code.code_name = f'{code.code_type}_K{code.k}_N{code.n}'
    
    channel = args.channel
    mode = args.mode
    diffu_step = args.diffu_step
    
    main(args)