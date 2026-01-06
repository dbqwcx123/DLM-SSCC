import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

def scan_and_calculate_size(src_dir, threshold=1080, fixed_width=2032, base_num=16):
    """
    第一步：扫描所有图片，统一按“横屏”逻辑筛选，计算统一高度。
    """
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not files:
        raise ValueError(f"在路径 {src_dir} 下没有找到图片文件")
        
    valid_files = []
    min_h = np.inf
    
    print(f"正在扫描并筛选图片 (自动旋转竖图 -> 横图)...")
    print(f"筛选条件: 短边 >= {threshold}, 长边(宽度) >= {fixed_width}")
    
    dropped_count = 0
    rotated_count = 0
    
    for filename in tqdm(files):
        src_path = os.path.join(src_dir, filename)
        try:
            with Image.open(src_path) as img:
                w, h = img.size
                
                # --- 核心修改：逻辑上的旋转 ---
                # 如果高度 > 宽度，我们在逻辑上交换它们，模拟旋转后的尺寸
                if h > w:
                    w, h = h, w
                    rotated_count += 1
                
                # 现在 w 必定是长边，h 必定是短边 (或者是正方形)
                
                # 1. 检查短边 (高度)
                if h < threshold:
                    # print(f"  [剔除] {filename}: 有效高度 {h} < 阈值 {threshold}")
                    dropped_count += 1
                    continue
                
                # 2. 检查长边 (宽度)
                if w < fixed_width:
                    # print(f"  [剔除] {filename}: 有效宽度 {w} < 目标宽度 {fixed_width}")
                    dropped_count += 1
                    continue
                
                # 记录合格图片
                valid_files.append(filename)
                
                # 更新最小高度 (基于旋转后的高度)
                if h < min_h:
                    min_h = h
                    
        except Exception as e:
            print(f"读取图片 {filename} 失败: {e}")
            
    if not valid_files:
        raise ValueError("没有找到任何符合条件的图片！")

    print(f"\n扫描结束：")
    print(f"  - 总文件数: {len(files)}")
    print(f"  - 需旋转图片数(竖图): {rotated_count}")
    print(f"  - 被剔除文件数(尺寸不足): {dropped_count}")
    print(f"  - 合格文件数: {len(valid_files)}")
    print(f"  - 合格图片中的最小高度: {min_h}")
    
    # 计算统一高度 (向下取整)
    unified_height = min_h - (min_h % base_num)
    
    return valid_files, unified_height

def process_and_save(src_dir, dst_dir, valid_files, target_w, target_h):
    """
    第二步：处理合格图片，执行旋转(如有必要)和裁剪
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"创建输出目录: {dst_dir}")
        
    print(f"\n开始处理 {len(valid_files)} 张合格图片...")
    print(f"目标统一尺寸: 宽 {target_w} x 高 {target_h}")
    
    success_count = 0
    
    for filename in tqdm(valid_files):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                w, h = img.size
                
                # --- 核心修改：实际执行旋转 ---
                # 如果是竖图，逆时针旋转90度变为横图
                if h > w:
                    img = img.transpose(Image.Transpose.ROTATE_90)
                    # 旋转后，w 和 h 会自动交换，但为了保险，我们只依赖 image 对象
                
                # 执行裁剪 (从左上角 0,0 开始)
                # 此时图像必定是横图，且尺寸已在第一步验证过足够大
                img_cropped = img.crop((0, 0, target_w, target_h))
                
                img_cropped.save(dst_path, quality=100)
                success_count += 1
                
        except Exception as e:
            print(f"保存图片 {filename} 时出错: {e}")
            
    return success_count

if __name__ == "__main__":
    # ---------------- 配置区域 ----------------
    source_path = "../Dataset/Kodak/Kodak24"      
    target_path = "../Dataset/Kodak/Kodak_unified" 
    
    base_num = 16          
    fixed_width = 768     
    min_threshold = 512   # 凡是短边小于此值的图(无论横竖)都会被丢弃
    # -----------------------------------------

    # 1. 检查源目录
    if not os.path.exists(source_path):
        print(f"错误：源目录不存在 -> {source_path}")
        sys.exit(1)

    try:
        # 2. 扫描、筛选并计算尺寸
        valid_files_list, unified_height = scan_and_calculate_size(
            source_path, 
            threshold=min_threshold, 
            fixed_width=fixed_width, 
            base_num=base_num
        )
        
        print(f"\n计算出的统一高度 (被16整除): {unified_height}")
        print(f"最终所有输出图像尺寸将为: {fixed_width} x {unified_height}")
        
        # 3. 执行处理
        count = process_and_save(source_path, target_path, valid_files_list, fixed_width, unified_height)
        
        print(f"\n========================================")
        print(f" 处理完成")
        print(f"========================================")
        print(f" 成功输出图像数量: {count}")
        print(f" 输出目录: {target_path}")
        print(f" 请更新 constants.py 中的 IMAGE_SHAPE 为: ({unified_height}, {fixed_width}, 3)")
        
    except ValueError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")