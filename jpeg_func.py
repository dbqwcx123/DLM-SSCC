import numpy as np
from PIL import Image
import io
from . import constants

def compress_jpeg(data: np.ndarray, quality: int = 100):
    """
    使用JPEG算法压缩图像数据。

    Args:
        data (np.ndarray): 输入的图像数据 (H, W, C)。
        quality (int): JPEG压缩质量 (1-100)。

    Returns:
        tuple: (压缩后的字节流, None)
    """
    # 确保数据类型是 uint8，这是图像库所期望的
    data = data.astype(np.uint8)
    # PIL 对数组的形状有特定的要求，对于灰度图像，形状应该是 (height, width)
    if data.ndim == 3 and data.shape[2] == 1:
        data = np.squeeze(data, axis=2)  # 移除通道维度
    # 从numpy数组创建Pillow图像对象
    pil_image = Image.fromarray(data, mode='L')  # 'L' 表示灰度图像
    
    # 使用内存缓冲区保存JPEG
    with io.BytesIO() as output:
        pil_image.save(output, format="JPEG", quality=quality)
        compressed_bytes = output.getvalue()
    
    return compressed_bytes


def decompress_jpeg(compressed_bits: str) -> np.ndarray:
    """
    从比特流解压JPEG图像数据。

    Args:
        compressed_bits (str): 输入的压缩比特流字符串。

    Returns:
        np.ndarray: 解压后的图像数据 (H, W, 1)，以匹配模型输出的维度。
    """
    # 检查比特流长度是否为8的倍数
    if len(compressed_bits) % 8 != 0:
        raise ValueError("比特流的长度必须是8的倍数")
        
    try:
        # 将比特流字符串转换为字节数据
        byte_list = []
        for i in range(0, len(compressed_bits), 8):
            byte = compressed_bits[i:i+8]
            byte_list.append(int(byte, 2))
        compressed_bytes = bytes(byte_list)
        
        # 使用内存缓冲区从字节数据读取JPEG
        with io.BytesIO(compressed_bytes) as input_stream:
            pil_image = Image.open(input_stream)
            # 将Pillow图像转换为numpy数组
            reconstructed_patch = np.array(pil_image)
            h, w = constants.CHUNK_SHAPE_2D
            if reconstructed_patch.shape != (h, w):  # PIL的size是(width, height)
                raise ValueError(f"解压图像尺寸{reconstructed_patch.shape}不符合预期{(h, w)}")

        # 确保输出的数组是三维的 (H, W, 1)，以匹配LLM解压后的格式
        if reconstructed_patch.ndim == 2:
            reconstructed_patch = np.expand_dims(reconstructed_patch, axis=2)
            
        return reconstructed_patch
    except Exception as e:
        print(f"JPEG解压失败: {e}")
        # 返回一个黑色块作为错误处理
        h, w = constants.CHUNK_SHAPE_2D
        return np.full((h, w, 1), 0, dtype=np.uint8)