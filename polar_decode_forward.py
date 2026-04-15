from __future__ import print_function
import argparse
import os
import numpy as np
# --- 猴子补丁：为了兼容老版本 python-polar-coding 库 ---
np.int = int
# np.float = float
# np.bool = bool
# ----------------------------------------------------

import torch
import time
from tqdm import tqdm
import constants

# 保持与 MM_ECCT_forward.py 一致的工具导入
from utils.MM_ECCT_utils import *
import warnings
warnings.filterwarnings("ignore")


##################################################################
# 数据处理与恢复模块 (与 MM_ECCT 完全一致)
##################################################################

def read_message_from_txt(filename, code):
    message_list = []
    with open(filename, 'r') as f:
        message_string_list = f.read().splitlines()
        for message_string in message_string_list:
            if not message_string.strip(): continue
            message = [int(x) for x in message_string]
            message_list.append(message)
    
    message_len = sum(len(msg) for msg in message_list)
    codeword_len = sum((len(msg) + code.k - 1) // code.k * code.n for msg in message_list)
    return message_list, message_len, codeword_len

def split_message(message, k):
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
    return result

def reconstruct_lines(all_pred_bits, message_list, code, info_positions):
    reconstructed_lines = []
    bit_pointer = 0
    N, K = code.n, code.k
    
    for original_msg in message_list:
        msg_len = len(original_msg)
        num_blocks = (msg_len + K - 1) // K
        chunk = all_pred_bits[bit_pointer : bit_pointer + num_blocks * N]
        bit_pointer += num_blocks * N
        
        current_line_str = ""
        for i in range(0, len(chunk), code.n):
            block = chunk[i : i + code.n]
            info_bits = [block[idx] for idx in info_positions]
            current_line_str += "".join(map(str, info_bits))
            
        current_line_str = current_line_str.rstrip('0')            
        reconstructed_lines.append(current_line_str)
    return reconstructed_lines

##################################################################
# 核心解码算法模块
##################################################################

def get_polar_transform_matrix(n):
    """生成标准的 Polar 极化变换矩阵 F^{\otimes \log2(n)}"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    m = int(np.log2(n))
    F_n = F
    for _ in range(m - 1):
        F_n = np.kron(F_n, F)
    return F_n

def scl_decode_batch(llr_batch, code, list_size):
    """
    终极修正版 SCL Wrapper：通过代数逆向推导精确对齐码本空间
    """
    from python_polar_coding.polar_codes import SCListPolarCodec
    
    batch_size = llr_batch.shape[0]
    x_pred = np.zeros((batch_size, code.n), dtype=int)
    
    # -------------------------------------------------------------
    # 核心数学逆向：从你的 G 矩阵中推导出底层真实的逻辑极化 Mask
    # -------------------------------------------------------------
    G_mat = code.generator_matrix.transpose(0, 1).cpu().numpy().astype(int)
    F_n = get_polar_transform_matrix(code.n)
    
    # 计算逻辑基底 U_basis = G * F_n % 2
    U_basis = np.dot(G_mat, F_n) % 2
    
    # 寻找 U_basis 中非全零的列，这些就是极化码中真实的非冻结位！
    col_sums = np.sum(U_basis, axis=0)
    logical_info_indices = np.where(col_sums > 0)[0]
    
    # 组装为 python-polar-coding 支持的字符串格式 Mask (例如 "00010111...")
    mask_str = "".join(['1' if i in logical_info_indices else '0' for i in range(code.n)])
    
    # -------------------------------------------------------------
    # 初始化解码器，在初始化时通过 mask 参数注入真实的逻辑掩码
    # -------------------------------------------------------------
    codec = SCListPolarCodec(
        N=code.n, 
        K=code.k, 
        design_snr=0.0, 
        is_systematic=False, 
        mask=mask_str,
        L=list_size
    )
    
    for i in range(batch_size):
        # 传入 LLR，解码器在正确的子空间内输出 K 个逻辑位
        decoded_u_info = codec.decode(llr_batch[i]) 
        
        # 重新编码，完美还原为 N 比特物理码字
        codeword_pred = codec.encode(decoded_u_info)
        x_pred[i] = codeword_pred
        
    return x_pred

def min_sum_bp_decode(llr, H, max_iter):
    """
    纯 Python 实现的 Min-Sum BP (最小和置信度传播) 核心。
    基于校验矩阵 H 进行消息传递。
    """
    N, M = H.shape[1], H.shape[0] # N: 变量节点数, M: 校验节点数
    
    # 初始化消息矩阵: 变量节点到校验节点 (V2C), 校验节点到变量节点 (C2V)
    V2C = np.zeros((M, N))
    C2V = np.zeros((M, N))
    
    # 初始 V2C 消息等于接收到的 LLR
    for j in range(N):
        V2C[:, j] = llr[j] * H[:, j]
        
    for _ in range(max_iter):
        # 1. 更新校验节点到变量节点 (C2V)
        for i in range(M):
            connected_v = np.where(H[i, :] == 1)[0]
            for j in connected_v:
                # 排除当前节点 j 的信息
                other_v = [v for v in connected_v if v != j]
                if len(other_v) == 0: continue
                
                # Min-Sum 核心：符号相乘，绝对值取最小
                signs = np.sign(V2C[i, other_v])
                mags = np.abs(V2C[i, other_v])
                
                prod_sign = np.prod(signs) if len(signs) > 0 else 1.0
                min_mag = np.min(mags) if len(mags) > 0 else 0.0
                
                C2V[i, j] = prod_sign * min_mag
                
        # 2. 更新变量节点到校验节点 (V2C)
        for j in range(N):
            connected_c = np.where(H[:, j] == 1)[0]
            for i in connected_c:
                other_c = [c for c in connected_c if c != i]
                V2C[i, j] = llr[j] + np.sum(C2V[other_c, j])
                
    # 3. 最终判决：LLR + 所有连接的 C2V 消息
    llr_out = np.copy(llr)
    for j in range(N):
        connected_c = np.where(H[:, j] == 1)[0]
        llr_out[j] += np.sum(C2V[connected_c, j])
        
    # 硬判决：LLR < 0 时判决为 1，LLR >= 0 判决为 0
    decoded_bits = (llr_out < 0).astype(int)
    return decoded_bits


def bp_decode_batch(llr_batch, code, max_iter, num_permutations):
    """
    BP 框架实现。
    """
    batch_size = llr_batch.shape[0]
    x_pred = np.zeros((batch_size, code.n), dtype=int)
    
    H = code.pc_matrix.cpu().numpy() # 提取校验矩阵
    
    for i in range(batch_size):
        # TODO: 这里是标准 BP。
        # 如果要实现完整的 AR-BP，需要在外部定义 Polar 码的自同构置换矩阵组 (Permutation Matrices)。
        # 然后对 H 矩阵进行打乱并行解码，最后合并 LLR。
        # 为保证代码能通用运行，此处运行标准 Min-Sum BP。
        x_pred[i] = min_sum_bp_decode(llr_batch[i], H, max_iter)
        
    return x_pred

##################################################################
# 评估与主循环
##################################################################

def estimate(m_tensor, x_tensor, code, sigma, channel_type, args):
    total_blocks = len(m_tensor)
    all_x_pred_bits = []
    total_ber = 0.
    total_fer = 0.
    
    batch_size = args.test_batch_size
    
    for i in tqdm(range(0, total_blocks, batch_size), desc=f"{args.decode_algo} Inference", leave=False):
        m_batch = m_tensor[i : i + batch_size]
        x_batch = x_tensor[i : i + batch_size]
        current_batch_size = len(x_batch)
        
        # 信道加噪 (保持与 MM_ECCT 一致)
        z = np.random.randn(current_batch_size, code.n) * sigma
        if channel_type == 'AWGN':
            h = np.ones((current_batch_size, code.n))
        elif channel_type == 'Rayleigh':
            h = np.random.rayleigh(1, (current_batch_size, code.n))
        else:
            raise ValueError("Invalid channel type.")
        
        # 调制 (0 -> 1, 1 -> -1)
        x_modulated = 1.0 - 2.0 * x_batch.numpy() 
        y_batch = h * x_modulated + z
        
        # === 核心：计算对数似然比 LLR ===
        # LLR = 2 * y * h / sigma^2
        llr_batch = (2.0 * y_batch * h) / (sigma ** 2)
        
        # 根据算法选择解码器
        if args.decode_algo == 'SCL':
            x_pred = scl_decode_batch(llr_batch, code, args.list_size)
        elif args.decode_algo == 'BP':
            x_pred = bp_decode_batch(llr_batch, code, args.bp_iter, args.num_perms)
        else:
            raise ValueError(f"Unknown decoding algorithm: {args.decode_algo}")
        
        # 统计指标
        x_batch_np = x_batch.numpy().astype(int)
        errors = np.sum(x_pred != x_batch_np, axis=1)
        total_ber += np.sum(errors) / code.n
        total_fer += np.sum(errors > 0)
        
        all_x_pred_bits.extend(x_pred.flatten().tolist())
        
    avg_ber = total_ber / total_blocks
    avg_fer = total_fer / total_blocks
    
    return avg_ber, avg_fer, all_x_pred_bits

class Code():
    pass

def main(args):
    msg_dir = f'./image_io/{args.mode}/diffu_step{args.diffu_step}/{args.msg_filename}.txt'
    if not os.path.exists(msg_dir):
        raise FileNotFoundError(f"Message_dir {msg_dir} does not exist.")
    
    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    
    # 获取生成矩阵和校验矩阵 (与 MM_ECCT 一致)
    G, H, H2 = Get_Generator_and_Parity(code, standard_form=False)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H2).long()
    code.code_name = f'{code.code_type}_K{code.k}_N{code.n}'
    
    message_list, message_len, codeword_len = read_message_from_txt(msg_dir, code)
    print(f'Using Algorithm: {args.decode_algo}')
    print(f'Message total length: {message_len}, Codeword total length: {codeword_len}')
    
    # 预编码展平
    all_m = []
    for msg in message_list:
        groups = split_message(msg, code.k)
        all_m.extend(groups)
    m_tensor = torch.tensor(all_m, dtype=torch.float)
    x_tensor = torch.matmul(m_tensor, code.generator_matrix.transpose(0, 1).float()) % 2
    
    # SNR 计算逻辑
    SNR_range_test = list(range(0, 3))
    
    _out_channel = 2 * 8
    _float_base = 32
    codeword_len_unified = _out_channel * (constants.IMAGE_SHAPE_TEST[0]//4) * (constants.IMAGE_SHAPE_TEST[1]//4) * _float_base * constants.NUM_IMAGE_TEST
    SNR_range_test_real = SNR_range_test + 10 * np.log10(codeword_len_unified / codeword_len)
    std_test = [SNR_to_std(ii) for ii in SNR_range_test_real]
    
    # 寻找信息位位置
    G_mat = code.generator_matrix.transpose(0, 1).cpu().numpy()
    code.info_positions = [np.nonzero(G_mat[i])[0][0] for i in range(code.k)]
    
    final_ber, final_fer = [], []
    
    # 输出路径前缀：根据选择的算法动态调整
    output_prefix = "SCL_forward" if args.decode_algo == 'SCL' else "BP_forward"
    
    for ii, snr_val in enumerate(SNR_range_test_real):
        print(f"\n--- Testing SNR Index {ii}: {SNR_range_test[ii]} dB (Real: {snr_val:.2f} dB) ---")
        
        ber, fer, all_pred_bits = estimate(m_tensor, x_tensor, code, std_test[ii], args.channel, args)
        
        final_ber.append(ber)
        final_fer.append(fer)
        print(f"Result -> BER: {ber:.4e} | FER: {fer:.4e}")
        
        recovered_lines = reconstruct_lines(all_pred_bits, message_list, code, code.info_positions)
        
        output_extracted_filename = f'./image_io/{args.mode}/diffu_step{args.diffu_step}/{output_prefix}/{code.code_name}/{args.channel}/demo_decode_SNR_{SNR_range_test[ii]}.txt'
        os.makedirs(os.path.dirname(output_extracted_filename), exist_ok=True)
        
        with open(output_extracted_filename, 'w') as f:
            for line in recovered_lines:
                f.write(line + "\n")
                
    metric_filename = f'./image_io/{args.mode}/diffu_step{args.diffu_step}/{output_prefix}/{code.code_name}/{args.channel}/#demo_decode_metric.txt'
    os.makedirs(os.path.dirname(metric_filename), exist_ok=True)
    with open(metric_filename, "w") as f:
        f.write("BER:\n"+ ", ".join([f"{x:.8e}" for x in final_ber]))
        f.write("\nFER:\n"+ ", ".join([f"{x:.8e}" for x in final_fer]))
        
    print(f"\nTesting completed. Results saved to {output_prefix} directory.")

def get_args():
    parser = argparse.ArgumentParser(description='Polar Unified Decoding (SCL & BP)')
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--msg_filename', type=str, default='compress_output')
    
    # 核心控制参数
    parser.add_argument('--decode_algo', type=str, default='BP', choices=['SCL', 'BP'], help='Choose decoding algorithm')
    
    # 信道与编码参数
    parser.add_argument('--code_type', type=str, default='POLAR')
    parser.add_argument('--code_k', type=int, default=32)
    parser.add_argument('--code_n', type=int, default=64)
    parser.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh'])
    
    # 路径参数
    parser.add_argument('--mode', type=str, default='CIFAR10/patch(16, 16)/diffugpt-s_ddm-sft/train_20251226_231454')
    parser.add_argument('--diffu_step', type=int, default=10)
    
    # SCL 专属参数
    parser.add_argument('--list_size', type=int, default=8, help='List size for SCL algorithm')
    
    # AR-BP 专属参数
    parser.add_argument('--bp_iter', type=int, default=50, help='Max BP iterations')
    parser.add_argument('--num_perms', type=int, default=4, help='Number of parallel BP graphs for AR-BP')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    set_seed(7)
    main(args)