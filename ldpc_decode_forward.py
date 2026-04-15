from __future__ import print_function
import argparse
import os
import numpy as np

import torch
from tqdm import tqdm
import constants

from utils.MM_ECCT_utils import *
import warnings
warnings.filterwarnings("ignore")


##################################################################
# 数据处理与恢复模块 (与 polar_decode_forward.py 保持一致)
##################################################################

def read_message_from_txt(filename, code):
    message_list = []
    with open(filename, 'r') as f:
        message_string_list = f.read().splitlines()
        for message_string in message_string_list:
            if not message_string.strip():
                continue
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
        chunk = all_pred_bits[bit_pointer: bit_pointer + num_blocks * N]
        bit_pointer += num_blocks * N

        current_line_str = ""
        for i in range(0, len(chunk), code.n):
            block = chunk[i: i + code.n]
            info_bits = [block[idx] for idx in info_positions]
            current_line_str += "".join(map(str, info_bits))

        current_line_str = current_line_str[:msg_len]
        reconstructed_lines.append(current_line_str)
    return reconstructed_lines


##################################################################
# LDPC 解码核心模块
##################################################################

def build_tanner_graph(H):
    """预计算 Tanner graph 邻接表，避免迭代中重复 np.where。"""
    check_to_var = [np.where(H[m] == 1)[0] for m in range(H.shape[0])]
    var_to_check = [np.where(H[:, n] == 1)[0] for n in range(H.shape[1])]
    return check_to_var, var_to_check


def hard_decision_from_llr(llr):
    return (llr < 0).astype(np.int32)


def check_syndrome(bits, H):
    syndrome = (H @ bits) % 2
    return np.all(syndrome == 0)


def min_sum_decode(llr, H, check_to_var, var_to_check, max_iter, alpha=1.0):
    """
    Min-Sum / Normalized Min-Sum 解码。
    alpha=1.0 时为标准 Min-Sum；alpha<1 时为 normalized Min-Sum。
    """
    M, N = H.shape
    V2C = np.zeros((M, N), dtype=np.float64)
    C2V = np.zeros((M, N), dtype=np.float64)

    for n in range(N):
        for m in var_to_check[n]:
            V2C[m, n] = llr[n]

    posterior = np.array(llr, dtype=np.float64, copy=True)

    for _ in range(max_iter):
        # Check node update
        for m in range(M):
            neigh = check_to_var[m]
            if len(neigh) == 0:
                continue

            incoming = V2C[m, neigh]
            signs = np.sign(incoming)
            signs[signs == 0] = 1.0
            abs_vals = np.abs(incoming)

            total_sign = np.prod(signs)
            if len(neigh) == 1:
                C2V[m, neigh[0]] = 0.0
                continue

            min1_idx = np.argmin(abs_vals)
            min1 = abs_vals[min1_idx]
            masked = abs_vals.copy()
            masked[min1_idx] = np.inf
            min2 = np.min(masked)

            for idx, n in enumerate(neigh):
                min_ex = min2 if idx == min1_idx else min1
                C2V[m, n] = alpha * total_sign * signs[idx] * min_ex

        # Variable node update and posterior update
        posterior = np.array(llr, dtype=np.float64, copy=True)
        for n in range(N):
            checks = var_to_check[n]
            if len(checks) == 0:
                continue
            total_msg = np.sum(C2V[checks, n])
            posterior[n] += total_msg
            for m in checks:
                V2C[m, n] = posterior[n] - C2V[m, n]

        bits = hard_decision_from_llr(posterior)
        if check_syndrome(bits, H):
            return bits

    return hard_decision_from_llr(posterior)


def sum_product_decode(llr, H, check_to_var, var_to_check, max_iter, clip_value=50.0):
    """
    标准 BP / Sum-Product 解码（LLR 域）。
    """
    M, N = H.shape
    V2C = np.zeros((M, N), dtype=np.float64)
    C2V = np.zeros((M, N), dtype=np.float64)

    for n in range(N):
        for m in var_to_check[n]:
            V2C[m, n] = llr[n]

    posterior = np.array(llr, dtype=np.float64, copy=True)

    for _ in range(max_iter):
        # Check node update
        for m in range(M):
            neigh = check_to_var[m]
            if len(neigh) == 0:
                continue

            tanh_vals = np.tanh(np.clip(V2C[m, neigh], -clip_value, clip_value) / 2.0)
            tanh_vals = np.clip(tanh_vals, -0.999999, 0.999999)

            for idx, n in enumerate(neigh):
                if len(neigh) == 1:
                    C2V[m, n] = 0.0
                    continue
                other = np.delete(tanh_vals, idx)
                prod = np.prod(other) if len(other) > 0 else 0.0
                prod = np.clip(prod, -0.999999, 0.999999)
                C2V[m, n] = 2.0 * np.arctanh(prod)

        # Variable node update and posterior update
        posterior = np.array(llr, dtype=np.float64, copy=True)
        for n in range(N):
            checks = var_to_check[n]
            if len(checks) == 0:
                continue
            total_msg = np.sum(C2V[checks, n])
            posterior[n] += total_msg
            for m in checks:
                V2C[m, n] = posterior[n] - C2V[m, n]

        posterior = np.clip(posterior, -clip_value, clip_value)
        bits = hard_decision_from_llr(posterior)
        if check_syndrome(bits, H):
            return bits

    return hard_decision_from_llr(posterior)


def decode_batch(llr_batch, code, args):
    batch_size = llr_batch.shape[0]
    x_pred = np.zeros((batch_size, code.n), dtype=np.int32)

    H = code.pc_matrix.cpu().numpy().astype(np.int32)
    check_to_var, var_to_check = build_tanner_graph(H)

    for i in range(batch_size):
        if args.decode_algo == 'MSA':
            x_pred[i] = min_sum_decode(
                llr_batch[i], H, check_to_var, var_to_check,
                max_iter=args.bp_iter, alpha=args.msa_alpha
            )
        elif args.decode_algo == 'BP':
            x_pred[i] = sum_product_decode(
                llr_batch[i], H, check_to_var, var_to_check,
                max_iter=args.bp_iter, clip_value=args.llr_clip
            )
        else:
            raise ValueError(f"Unknown decoding algorithm: {args.decode_algo}")

    return x_pred


##################################################################
# 评估与主循环
##################################################################

def estimate(m_tensor, x_tensor, code, sigma, channel_type, args):
    total_blocks = len(m_tensor)
    all_x_pred_bits = []
    total_ber = 0.0
    total_fer = 0.0

    batch_size = args.test_batch_size

    for i in tqdm(range(0, total_blocks, batch_size), desc=f"{args.decode_algo} Inference", leave=False):
        m_batch = m_tensor[i: i + batch_size]
        x_batch = x_tensor[i: i + batch_size]
        current_batch_size = len(x_batch)

        z = np.random.randn(current_batch_size, code.n) * sigma
        if channel_type == 'AWGN':
            h = np.ones((current_batch_size, code.n))
        elif channel_type == 'Rayleigh':
            h = np.random.rayleigh(1, (current_batch_size, code.n))
        else:
            raise ValueError("Invalid channel type.")

        x_modulated = 1.0 - 2.0 * x_batch.numpy()
        y_batch = h * x_modulated + z

        llr_batch = (2.0 * y_batch * h) / (sigma ** 2)
        llr_batch = np.clip(llr_batch, -args.llr_clip, args.llr_clip)

        x_pred = decode_batch(llr_batch, code, args)

        x_batch_np = x_batch.numpy().astype(np.int32)
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

    G, H, H2 = Get_Generator_and_Parity(code, standard_form=False)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H2).long()
    code.code_name = f'{code.code_type}_K{code.k}_N{code.n}'

    message_list, message_len, codeword_len = read_message_from_txt(msg_dir, code)
    print(f'Using Algorithm: {args.decode_algo}')
    print(f'Message total length: {message_len}, Codeword total length: {codeword_len}')

    all_m = []
    for msg in message_list:
        groups = split_message(msg, code.k)
        all_m.extend(groups)
    m_tensor = torch.tensor(all_m, dtype=torch.float)
    x_tensor = torch.matmul(m_tensor, code.generator_matrix.transpose(0, 1).float()) % 2

    SNR_range_test = list(range(0, 6))

    _out_channel = 2 * 8
    _float_base = 32
    codeword_len_unified = _out_channel * (constants.IMAGE_SHAPE_TEST[0] // 4) * (constants.IMAGE_SHAPE_TEST[1] // 4) * _float_base * constants.NUM_IMAGE_TEST
    SNR_range_test_real = SNR_range_test + 10 * np.log10(codeword_len_unified / codeword_len)
    std_test = [SNR_to_std(ii) for ii in SNR_range_test_real]

    G_mat = code.generator_matrix.transpose(0, 1).cpu().numpy()
    code.info_positions = [np.nonzero(G_mat[i])[0][0] for i in range(code.k)]

    final_ber, final_fer = [], []

    output_prefix = 'MSA_forward' if args.decode_algo == 'MSA' else 'BP_forward'

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
    with open(metric_filename, 'w') as f:
        f.write('BER:\n' + ', '.join([f'{x:.8e}' for x in final_ber]))
        f.write('\nFER:\n' + ', '.join([f'{x:.8e}' for x in final_fer]))

    print(f"\nTesting completed. Results saved to {output_prefix} directory.")


def get_args():
    parser = argparse.ArgumentParser(description='LDPC Unified Decoding (MSA & BP)')
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--msg_filename', type=str, default='compress_output')

    parser.add_argument('--decode_algo', type=str, default='MSA', choices=['MSA', 'BP'], help='Choose decoding algorithm')

    parser.add_argument('--code_type', type=str, default='LDPC')
    parser.add_argument('--code_k', type=int, default=24)
    parser.add_argument('--code_n', type=int, default=49)
    parser.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh'])

    parser.add_argument('--mode', type=str, default='CIFAR10/patch(16, 16)/diffugpt-s_ddm-sft/train_20251226_231454')
    parser.add_argument('--diffu_step', type=int, default=10)

    parser.add_argument('--bp_iter', type=int, default=50, help='Max iteration for iterative decoding')
    parser.add_argument('--msa_alpha', type=float, default=1.0, help='Scaling factor for normalized Min-Sum')
    parser.add_argument('--llr_clip', type=float, default=50.0, help='Numerical clipping value for LLR messages')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    set_seed(7)
    main(args)
