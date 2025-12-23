from __future__ import annotations

import functools
import os.path
from dataclasses import dataclass, field
from typing import Callable, Optional, Iterator, Union
import numpy as np
import einops
import torch
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
from matplotlib import pyplot as plt
import time
from collections import Counter, deque
from PIL import Image
from torchvision.transforms import Compose
from transformers import ImageGPTForCausalImageModeling, ImageGPTConfig
import sys
from tqdm import tqdm

# import constants
# import data_loaders
from .. import constants, data_loaders
from ..utils import arithmetic_coder
from ..utils.ac_utils import normalize_pdf_for_arithmetic_coding, bits_to_bytes, bytes_to_bits

import pdb
from collections import Counter
from thop import profile


model = None

device_id = 0
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


def calculate_flops(model: ImageGPTForCausalImageModeling, seq_len: int, device: torch.device):
    """
    使用 thop 库计算 ImageGPT 模型在自回归生成过程中的总 FLOPs。

    由于压缩过程是自回归的，模型会被调用 seq_len 次，
    每次输入的序列长度从 1 递增到 seq_len。
    此函数会累加每一次调用的计算量。

    Args:
        model: ImageGPT 模型实例。
        seq_len: 输入序列的总长度。
        device: 模型所在的设备 (e.g., 'cuda:0' or 'cpu')。

    Returns:
        整个自回归压缩过程的总计算量 (GFLOPs)。
    """
    total_macs = 0
    model.eval()  # 设置为评估模式

    # 循环模拟自回归过程
    for i in range(1, seq_len + 1):
        # 创建一个长度为 i 的虚拟输入
        dummy_input = torch.randint(0, model.config.vocab_size, (1, i), dtype=torch.long).to(device)

        # 使用 thop.profile 计算单次前向传播的 MACs
        # verbose=False 可以避免打印每一层的详细信息
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        total_macs += macs

    # FLOPs 通常约等于 2 * MACs
    total_flops = total_macs * 2

    # 将结果转换为 GFLOPs (1 GFLOP = 10^9 FLOPs)
    return total_flops / 1e9


def _retrieve_model_params(model_type, model_path=Union[str, functools.partial], ):
    global model
    if not isinstance(model, torch.nn.Module):
        if model_type=="igpt":
            # model_path = './data/model/igpt/igpt'
            model = ImageGPTForCausalImageModeling.from_pretrained(model_path)
            model.to(device)
    return model


def _retrieve_predict_fn(
        model: torch.nn.Module
) -> Callable[[np.ndarray], np.ndarray]:
    """Returns the prediction function for the trained model."""

    def get_pdf(_input):
        """Returns the probability distribution of the next token."""
        with torch.no_grad():
            # _input shape: (batch_size, seq_len)
            _input = torch.tensor(_input, dtype=torch.int64).to(device)
            # output['logits'] shape: (batch_size, seq_len, vocab_size)
            output = model(_input, output_hidden_states=True)
        gen_sequence_probs = output['logits'].squeeze().softmax(-1).cpu().detach().numpy()
        return gen_sequence_probs
    
    return get_pdf


def probs_normalization(probs, prob_value, index):
    current_last_prob = probs[index, -1]
    remaining_prob = 1 - prob_value
    # 计算其他元素的缩放因子
    scaling_factor = remaining_prob / (1 - current_last_prob)
    # 调整其他元素的概率
    probs[index, :-1] *= scaling_factor
    # 设置新的概率值
    probs[index, -1] = prob_value
    return probs


# 关键函数，用模型实现压缩
def compress(
        data,
        use_slow_lossless_compression: bool = False,
        model_path = None,
        patch_id = 0
) -> bytes | tuple[bytes, int] | tuple[bytes, np.ndarray]:
    """Compresses the `data` using arithmetic coding and a pretrained model.

  Args:
    data: The data to be compressed. => 形状为 (32, 32, 1) 的 numpy 数组（单通道图像块）
    use_slow_lossless_compression: Whether to compute the `pdf`s for all tokens
      in the data stream in one go or separately for every proper subsequence.
      When only compressing data (i.e., without decompression) use the first
      approach (i.e., `True`) since it has an O(n) runtime complexity, while the
      latter is O(n^2). However, the goal is to losslessly decompress the
      compressed output, use the second option (i.e., `False`) since this is
      what happens in the decoder (which iteratively reconstructs the sequence).
  Returns:
    The compressed data.
  """
    t0 = time.perf_counter()
    model_type = 'igpt'
    params = _retrieve_model_params(model_type, model_path)
    
    # 使用 thop 计算 FLOPs
    total_flops = calculate_flops(params, constants.CHUNK_SIZE_BYTES, device)
    
    predict_fn = _retrieve_predict_fn(params)
    
    t1 = time.perf_counter()
    sequence_array = data  # (h,w,1)
    # 把sequence_array输出到文件里记录
    # np.savetxt(f"./data/data/images/save_temp/sequence_array1_{patch_id}.csv", sequence_array.squeeze(), delimiter=',', fmt='%d')
    if isinstance(sequence_array, tuple):
        previous_array, sequence_array = data
    else:
        previous_array = sequence_array

    if not len(sequence_array.shape) > 3:
        # 说明是多通道压缩
        if model_type == 'igpt':
            test1 = Counter(sequence_array.reshape(-1))
            # print(f"\nraw count: {test1}, length: {len(test1)}\n")
            # 将图像块数据展平为一维序列，(h, w, 1) => (1, h*w)
            sequence_array = torch.from_numpy(sequence_array.reshape(-1)).to(device).to(torch.int64).reshape(1, constants.CHUNK_SIZE_BYTES)
    use_slow_lossless_compression = True  # 慢速压缩，逐个token预测
    if use_slow_lossless_compression:
        log_probs = list()
        subsequence_probs = predict_fn(torch.zeros((1, 1), dtype=torch.int64).to(device))
        log_probs.append(subsequence_probs)
        for subsequence_length in tqdm(range(constants.CHUNK_SIZE_BYTES)):
            subsequence_probs = predict_fn(
                sequence_array[:, : subsequence_length + 1]
            )
            if len(subsequence_probs.shape) < 2:
                log_probs.append(subsequence_probs)
            else:
                log_probs.append(subsequence_probs[-1])
            # print(subsequence_length)
        # pdb.set_trace()

        log_probs = np.vstack(log_probs)
    else:
        # 一次性直接预测对数概率分布
        # log_probs = predict_fn(sequence_array[None])
        #TODO 进行的改动,sequence_array没有经过tokenizer，现在是字符串
        log_probs = predict_fn(sequence_array[0])
        # 单独处理第一个 token 的概率，使用与 decompress 完全相同的技巧，确保概率一致
        beginning_prob = predict_fn(torch.zeros((1, 1), dtype=torch.int64).to(device))
        beginning_prob = beginning_prob.reshape(1,-1)
        log_probs = np.concatenate((beginning_prob, log_probs), axis=0)
    raw_probs = log_probs.reshape(-1, log_probs.shape[-1])
    t2 = time.perf_counter()
    output = list()
    
    # 算术码编码器
    encoder = arithmetic_coder.Encoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        output_fn=output.append,  # 输出位流到列表
    )

    # sequence_array = sequence_array.reshape(-1)
    sequence_array = sequence_array.detach().view(-1).cpu().numpy().squeeze()
    # max_indices = np.argmax(raw_probs, axis=1)
    # pdb.set_trace()
    # predict_right = sequence_array - max_indices
    # zeros = np.asarray(np.where(predict_right == 0)).squeeze()
    # print(f"predict right rate: {len(zeros) / len(sequence_array)}")
    
    # pdb.set_trace()
    symbols = []
    correct_symbols = []
    correct_pdf = []
    counter = 0
    
    for pdf, symbol in zip(raw_probs, sequence_array):
        symbols.append(symbol)
        if max(pdf) == pdf[symbol]:
            correct_symbols.append(symbol)
        correct_pdf.append(pdf[symbol])
        encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol)
        counter += 1
    # print(len(right_symbols))
    # print(len(right_symbols) / len(symbols))
    # print(right_symbols)

    # 测试编码
    #
    t4 = time.perf_counter()
    # print(f"本次压缩时间:{t4 - t3}")
    encoder.terminate()
    # 将编码后的bit转换为字符串类型
    compressed_bits = ''.join(map(str, output))
    # 将bit位转换为字节bytes
    # compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)

    t5 = time.perf_counter()
    # print(f"time cost:{t5 - t0}")

    # test the decoder
    # decompress(compressed_bytes, 0, len(sequence_array))
    return compressed_bits, log_probs, total_flops


count = 0


def decompress(
        data,
        uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
        model_path = None
) -> np.ndarray:
    """Decompresses the `data` using arithmetic coding and a pretrained model.

  See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

  Args:
    data: The data to be decompressed.
    uncompressed_length: The length of the original data stream (in bytes).

  Returns:
    The decompressed image patch as a numpy array of shape (H, W, C).
  """
    # 设定一个阈值，当连续N次解码出同一个token时，视为解码失败
    CONSECUTIVE_REPEAT_THRESHOLD = 5  # 您可以根据需要调整此值
    
    # 1. 初始化模型和解码器
    model_type = 'igpt'
    params = _retrieve_model_params(model_type, model_path)
    predict_fn = _retrieve_predict_fn(params)
    
    data_iter = iter(data)

    # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
    # from the compressed input and returns `None` when the input is exhausted.
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
        try:
            global count
            count += 1
            # print(count)
            return int(next(bit_sequence))
        except StopIteration:
            return None

    # 算术码解码器
    decoder = arithmetic_coder.Decoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        input_fn=_input_fn,
    )

    # 2. 预分配一个用0填充的完整序列
    sequence_array = torch.zeros((1, uncompressed_length), dtype=torch.int64).to(device)
    # 使用双端队列高效地跟踪最近的token
    # recent_tokens = deque(maxlen=CONSECUTIVE_REPEAT_THRESHOLD)
    
    # 3. 递归生成，迭代解码
    for idx in tqdm(range(uncompressed_length)):
        # a. 基于当前已解码的序列，预测下一个 token 的概率分布
        # 只将已生成的序列部分传递给模型
        current_sequence = sequence_array[:, :idx]
        if current_sequence.shape[1] == 0:
            # 对于第一个token，输入一个空的上下文
            subsequence_probs = predict_fn(torch.zeros((1, 1), dtype=torch.int64).to(device))
        else:
            # 传递当前已解码的序列
            subsequence_probs = predict_fn(current_sequence)
        
        if len(subsequence_probs.shape) < 2:
            probs = subsequence_probs
        else:
            probs = subsequence_probs[-1]
        
        # b. 使用解码器和概率分布，从比特流中解码出下一个 token
        try:
            decoded_token = decoder.decode(normalize_pdf_for_arithmetic_coding(probs))
        except Exception as e:
            print(f"\n警告: 在第 {idx+1} 个 token 处解码失败，提前终止。错误: {e}")
            break  # 退出循环，剩余部分将保持为0
        
        # c. 将解码出的token放入预分配的数组中
        # new_token = torch.tensor([[decoded_token]], dtype=torch.int64).to(device)
        # sequence_array = torch.cat((sequence_array, new_token), dim=1)
        sequence_array[0, idx] = decoded_token
        
        # d. 检测连续重复的 token
        # recent_tokens.append(decoded_token)
        # if len(set(recent_tokens)) == 1 and len(recent_tokens) == CONSECUTIVE_REPEAT_THRESHOLD:
        #     print(f"\n警告: 在第 {idx+1} 个 token 处解码失败，连续 {CONSECUTIVE_REPEAT_THRESHOLD} 次解码出相同的 token。")
        #     break  # 退出循环，剩余部分将保持为0
        
    # 4. 将解码后的 token 序列转换回图像 patch 的形状
    h, w = constants.CHUNK_SHAPE_2D
    reconstructed_patch = sequence_array.cpu().numpy().reshape(h, w, 1)

    return reconstructed_patch


def patch_visulaize(patch_data, save_path, idx):
    """
    将图像块可视化并保存为灰度图。

    参数:
    patch_data (np.ndarray): 形状为 (32, 32) 或 (32, 32, 1) 的图像块数据。
    save_path (str): 保存路径。
    idx (int): 图像块索引。
    """
    patch_data = np.asarray(patch_data)
    if patch_data.ndim == 3 and patch_data.shape[2] == 1:
        patch_data = patch_data[:, :, 0]
    if patch_data.ndim != 2:
        raise ValueError("patch_data 必须是 (32, 32) 的灰度图像")
    
    patch_data = patch_data.astype(np.uint8)

    # 使用 PIL 保存灰度图像
    img = Image.fromarray(patch_data, mode='L')  # 'L' 表示灰度图像
    img.save(f'{save_path}/patch{idx}.png')
    plt.close()


if __name__ == '__main__':
    model_path = './data/model/igpt/igpt'
    data_path = './data/data/images/ILSVRC_temp'
    save_dir = './data/data/images/save_temp/patches'
    os.makedirs(save_dir, exist_ok=True)
    
    data_iterator = data_loaders.get_imagenet_iterator(
                    num_chunks=constants.NUM_CHUNKS,
                    is_channel_wised=True,
                    is_seq=True,
                    data_path=data_path)
    
    for data, frame_id in data_iterator:
        patch_visulaize(data, save_dir+'/original', frame_id)
        
        compressed_bits, log_probs = compress(data, frame_id)
        compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)
        print(f"压缩后字节数:{len(compressed_bytes)}, 压缩后比特数:{len(compressed_bits)}")
        
        reconstructed_data = decompress(compressed_bits)
        
        patch_visulaize(reconstructed_data, save_dir+'/reconstructed', frame_id)
        
        # 比较原始和重建图像
        is_identical = np.array_equal(data, reconstructed_data)
        print(f"图像块 {frame_id} {'完全还原' if is_identical else '有差异'}")
        
        # 如果有差异，计算均方误差
        if not is_identical:
            mse = np.mean((data - reconstructed_data)**2)
            print(f"均方误差: {mse}")
