import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

def scan_and_filter(src_dir, min_short_side=1024):
    """
    第一步：扫描所有图片，筛选符合条件的图片。
    修改点：不再计算动态高度，而是严格筛选短边是否达标。
    """
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not files:
        raise ValueError(f"在路径 {src_dir} 下没有找到图片文件")
        
    valid_files = []
    
    print(f"正在扫描并筛选图片...")
    print(f"筛选条件: 短边 >= {min_short_side}")
    
    dropped_count = 0
    rotated_count = 0
    
    for filename in tqdm(files):
        src_path = os.path.join(src_dir, filename)
        try:
            with Image.open(src_path) as img:
                w, h = img.size
                
                # 逻辑旋转：确保 w 是长边，h 是短边
                if h > w:
                    w, h = h, w
                    rotated_count += 1
                
                # 修改点：直接检查短边是否小于固定阈值 (1024)
                if h < min_short_side:
                    dropped_count += 1
                    continue
                
                # 长边不做严格剔除，只要短边达标，长边不够的我们后面会填充
                
                valid_files.append(filename)
                    
        except Exception as e:
            print(f"读取图片 {filename} 失败: {e}")
            
    if not valid_files:
        raise ValueError("没有找到任何符合条件的图片！")

    print(f"\n扫描结束：")
    print(f"  - 总文件数: {len(files)}")
    print(f"  - 需旋转图片数(竖图): {rotated_count}")
    print(f"  - 被剔除文件数(短边 < {min_short_side}): {dropped_count}")
    print(f"  - 合格文件数: {len(valid_files)}")
    
    return valid_files

def process_and_save(src_dir, dst_dir, valid_files, target_w, target_h):
    """
    第二步：处理合格图片，执行旋转 -> 宽度填充 -> 高度裁剪
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"创建输出目录: {dst_dir}")
        
    print(f"\n开始处理 {len(valid_files)} 张合格图片...")
    print(f"目标固定尺寸: 宽 {target_w} x 高 {target_h}")
    print(f"填充策略: 镜像填充 (Reflection Padding)")
    
    success_count = 0
    
    for filename in tqdm(valid_files):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                w, h = img.size
                
                # 1. 旋转：统一为横图 (宽 > 高)
                if h > w:
                    img = img.transpose(Image.Transpose.ROTATE_90)
                    w, h = img.size # 更新尺寸
                
                # 2. 修改点：宽度填充 (Padding)
                # 针对 DIV2K (2040 -> 2048) 或其他宽度不足的情况
                if w < target_w:
                    # 计算差值
                    pad_total = target_w - w
                    # 左右各填一半
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                    
                    # 转换为 Numpy 数组进行填充
                    img_np = np.array(img)
                    
                    # pad_width 格式: ((top, bottom), (left, right), (channels, channels))
                    # mode='reflect' 即镜像填充，效果最好
                    img_np = np.pad(
                        img_np, 
                        ((0, 0), (pad_left, pad_right), (0, 0)), 
                        mode='reflect'
                    )
                    
                    # 转回 PIL 对象
                    img = Image.fromarray(img_np)
                
                # 3. 中心裁剪 (Center Crop)
                # 务必重新获取尺寸，因为上面可能进行过填充操作改变了 w, h
                current_w, current_h = img.size
                
                # 计算左上角坐标 (left, top)
                # 使用 // 2 确保是整数
                left = (current_w - target_w) // 2
                top = (current_h - target_h) // 2
                
                # 计算右下角坐标
                right = left + target_w
                bottom = top + target_h
                
                # 执行裁剪
                img_cropped = img.crop((left, top, right, bottom))
                
                img_cropped.save(dst_path, quality=100)
                success_count += 1
                
        except Exception as e:
            print(f"保存图片 {filename} 时出错: {e}")
            
    return success_count

if __name__ == "__main__":
    # ---------------- 配置区域 ----------------
    source_path = "../Dataset/DIV2K/DIV2K_valid_LR/X4"
    target_path = "../Dataset/DIV2K/DIV2K_LR_unified/valid"
    
    # 修改点：固定尺寸配置
    FIXED_WIDTH = 512    # 目标长边
    FIXED_HEIGHT = 256   # 目标短边
    MIN_THRESHOLD = 256  # 短边剔除阈值
    # -----------------------------------------

    # 1. 检查源目录
    if not os.path.exists(source_path):
        print(f"错误：源目录不存在 -> {source_path}")
        sys.exit(1)

    try:
        # 2. 扫描与筛选 (不再计算 unified_height，直接使用固定值)
        valid_files_list = scan_and_filter(
            source_path, 
            min_short_side=MIN_THRESHOLD
        )
        
        print(f"\n最终所有输出图像尺寸将为: {FIXED_WIDTH} x {FIXED_HEIGHT}")
        
        # 3. 执行处理
        count = process_and_save(source_path, target_path, valid_files_list, FIXED_WIDTH, FIXED_HEIGHT)
        
        print(f"\n========================================")
        print(f" 处理完成")
        print(f"========================================")
        print(f" 成功输出图像数量: {count}")
        print(f" 输出目录: {target_path}")
        print(f" 请更新 constants.py 中的 IMAGE_SHAPE 为: ({FIXED_HEIGHT}, {FIXED_WIDTH}, 3)")
        
    except ValueError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")