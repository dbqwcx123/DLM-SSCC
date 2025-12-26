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

##################################################################

class My_Dataset(data.Dataset):
    def __init__(self, message, code, sigma):
        self.message = message
        self.code = code
        self.sigma = sigma
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

    def __getitem__(self, index):
        num_groups, message_list = split_message(self.message, self.code.k)
        m_list, x_list, z_list, y_list, magnitude_list, syndrome_list = [], [], [], [], [], []
        
        for i in range(num_groups):
            single_msg = message_list[i]
            
            m = torch.tensor(single_msg).view(1, self.code.k)
            x = torch.matmul(m, self.generator_matrix) % 2
            
            z = torch.randn(self.code.n) * self.sigma[0]
            
            if channel == 'AWGN':
                h = 1
            elif channel == 'Rayleigh':
                h = torch.from_numpy(np.random.rayleigh(1, self.code.n)).float()
            else:
                raise ValueError("Invalid channel type.")
            
            y = h * bin_to_sign(x) + z
            magnitude = torch.abs(y)
            syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                    self.pc_matrix) % 2
            syndrome = bin_to_sign(syndrome)
            
            m_list.append(m.float())
            x_list.append(x.float())
            z_list.append(z.float())
            y_list.append(y.float())
            magnitude_list.append(magnitude.float())
            syndrome_list.append(syndrome.float())
            
        return m_list, x_list, z_list, y_list, magnitude_list, syndrome_list

    def __len__(self):
        return 1

##################################################################

def read_message_from_txt(filename):
    """读取消息文件并返回消息列表和统计信息"""
    message_list = []
    with open(filename, 'r') as f:
        message_string_list = f.read().splitlines()
        for message_string in message_string_list:
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
    return num_groups, result

def extract_information_bits(codeword_list, K, N):
    """从码字中提取信息比特"""
    extracted_messages = []
    
    for bitstream in codeword_list:
        decoded_bitstream = ""
        # 按码字长度分段处理
        for i in range(0, len(bitstream), N):
            segment = bitstream[i:i+N]
            if len(segment) < N:
                # 处理不完整的段
                segment = segment + [0] * (N - len(segment))
            
            # 提取前K个信息比特
            m = segment[:K]
            decoded_segment = "".join(map(str, m))
            decoded_bitstream += decoded_segment
        
        # 去除末尾的填充零
        decoded_bitstream = decoded_bitstream.rstrip('0')
        extracted_messages.append(decoded_bitstream)
    
    return extracted_messages

##################################################################

def estimate(model, device, dataloader_list, SNR_range_test, code):
    """估计性能并提取信息（整合了估计和信息提取功能）"""
    model.eval()
    test_loss_noise_ber, test_loss_list, test_loss_ber_list, test_loss_fer_list = [], [], [], []
    cum_samples_all = []
    
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(dataloader_list):
            (m_list, x_list, z_list, y_list, magnitude_list, syndrome_list) = next(iter(test_loader))
            
            noise_ber = test_loss = test_ber = test_fer = cum_count = 0.
            x_pred_list = []
            all_x_pred_bits = []  # 收集所有预测的比特
            
            for jj in range(len(m_list)):
                m, x, z, y, magnitude, syndrome = m_list[jj], x_list[jj], z_list[jj], y_list[jj], magnitude_list[jj], syndrome_list[jj]
                noise_ber += BER((y<=0).float(), x)
                z_mul = (y * bin_to_sign(x))
                
                z_pred = model(magnitude.to(device), syndrome.to(device))
                loss, x_pred = model.loss(z_pred, z_mul.to(device), y.to(device))

                test_loss += loss.item() * x.shape[0]
                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                
                # 收集预测结果
                x_pred_np = x_pred.squeeze(0).cpu().numpy().astype(int)
                x_pred_list.append(x_pred_np)
                all_x_pred_bits.extend(x_pred_np.flatten().tolist())
            
            # 保存性能指标
            cum_samples_all.append(cum_count)
            test_loss_noise_ber.append(noise_ber / cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            
            # 直接提取信息比特并保存
            extracted_messages = extract_information_bits([all_x_pred_bits], code.k, code.n)
            
            # 保存提取后的信息
            output_extracted_filename = f'./image_io/{mode}/diffu_step{diffu_step}/ECCT_forward/{code.code_name}/{channel}/demo_decode_SNR_{SNR_range_test[ii]}.txt'
            os.makedirs(os.path.dirname(output_extracted_filename), exist_ok=True)
            with open(output_extracted_filename, 'a') as f:
                for msg in extracted_messages:
                    f.write(msg + "\n")
            
            # 原始预测结果
            # output_pred_filename = f'./image_io/{mode}/ECCT_pred/{code.code_name}/{channel}/demo_decode_SNR_{SNR_range_test[ii]}.txt'
            # os.makedirs(os.path.dirname(output_pred_filename), exist_ok=True)
            # with open(output_pred_filename, 'a') as f:
            #     combined_x_pred = np.concatenate(x_pred_list)
            #     f.write("".join(map(str, combined_x_pred)) + "\n")
        
        # 打印性能结果
        print('\nNoise BER ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr) in zip(test_loss_noise_ber, SNR_range_test)]))
        print('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr) in zip(test_loss_ber_list, SNR_range_test)]))
        print('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr) in zip(test_loss_fer_list, SNR_range_test)]))
    
    print(f'Test Time {time.time() - t:.2f} s\n')
    return test_loss_list, test_loss_ber_list, test_loss_fer_list

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
    
    # 日志配置
    # handlers = [logging.FileHandler(os.path.join(model_dir, 'logging_test.txt')), logging.StreamHandler()]
    # logging.basicConfig(level=print, format='%(message)s', handlers=handlers)
    
    print(f"Path to model\logs: {model_dir}")
    print(args)
    
    # 读取消息
    message_list, message_len, codeword_len = read_message_from_txt(args.msg_path)
    print(f'Message total length: {message_len}')
    print(f'Codeword total length: {codeword_len}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    if args.isParallel:
        model = torch.load(os.path.join(args.model_path, 'best_model'), map_location='cpu')
        model = model.module.to(device)
    else:
        model = torch.load(os.path.join(args.model_path, 'best_model'))
        model.to(device)
    
    print(f'Transmission channel type: {args.channel}')
    
    # SNR范围设置
    SNR_range_test = np.arange(0, 8)
    ## TODO: 这里需要根据baseline调整SNR范围
    codeword_len_unified = 16*8*8*32*10
    print(f'Codeword length unified: {codeword_len_unified}')
    SNR_range_test_real = SNR_range_test + 10*np.log10(codeword_len_unified/codeword_len)
    # SNR_range_test_real = SNR_range_test
    print(f'SNR_range_test: {SNR_range_test_real}')
    
    std_test = [SNR_to_std(ii) for ii in SNR_range_test_real]
    print(f'std_test: {std_test}')
    
    # 性能评估
    test_loss_sum, test_loss_ber_sum, test_loss_fer_sum = [0] * len(std_test), [0] * len(std_test), [0] * len(std_test)
    
    for message in message_list:
        dataloader_list = [My_Dataset(message, code, sigma=[std_test[ii]]) for ii in range(len(std_test))]
        test_loss_list, test_loss_ber_list, test_loss_fer_list = estimate(
            model, device, dataloader_list, SNR_range_test, code)
        
        test_loss_sum = [x + y for x, y in zip(test_loss_list, test_loss_sum)]
        test_loss_ber_sum = [x + y for x, y in zip(test_loss_ber_list, test_loss_ber_sum)]
        test_loss_fer_sum = [x + y for x, y in zip(test_loss_fer_list, test_loss_fer_sum)]
    
    # 计算平均性能指标
    test_loss_avg = [x / len(message_list) for x in test_loss_sum]
    test_loss_ber_avg = [x / len(message_list) for x in test_loss_ber_sum]
    test_loss_fer_avg = [x / len(message_list) for x in test_loss_fer_sum]
    
    # 保存性能指标
    metric_filename = f'./image_io/{mode}/diffu_step{diffu_step}/ECCT_forward/{code.code_name}/{channel}/#demo_decode_metric.txt'
    os.makedirs(os.path.dirname(metric_filename), exist_ok=True)
    with open(metric_filename, "a") as f:
        f.write("\nloss:\n"+ ", ".join([f"{x:.8e}" for x in test_loss_avg]))
        f.write("\nBER:\n"+ ", ".join([f"{x:.8e}" for x in test_loss_ber_avg]))
        f.write("\nFER:\n"+ ", ".join([f"{x:.8e}" for x in test_loss_fer_avg]))
        
    print(f"Testing completed. Results save to {args.mode}")

##################################################################
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ECCT')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default="0", help='gpus ids')
    parser.add_argument('--test_batch_size', type=int, default=2048)

    # Message args
    parser.add_argument('--msg_filename', type=str, default='compressed_output')

    # Code args
    parser.add_argument('--code_type', type=str, default='LDPC', choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=24)
    parser.add_argument('--code_n', type=int, default=49)
    parser.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--mode', type=str, default='entropy_confidence/smooth_k0_alpha0/channel_corre/patch(16, 16)/diffugpt-s_ddm-sft/train_ckp-4000_251225')
    parser.add_argument('--diffu_step', type=int, default=100)

    # Model args
    parser.add_argument('--isParallel', type=bool, default=True)
    
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