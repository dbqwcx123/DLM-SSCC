import os
from PIL import Image
from tqdm import tqdm

def resize_and_save_images(src_dir, dst_dir, base_num=16):
    """
    将 src_dir 下的图片裁剪为 base_num 的整数倍，并保存到 dst_dir
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"创建输出目录: {dst_dir}")

    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"开始处理 {len(files)} 张图片...")
    
    for filename in tqdm(files):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB') # 确保统一为RGB
                w, h = img.size
                
                # 计算新尺寸（向下取整）
                new_w = w - (w % base_num)
                new_h = h - (h % base_num)
                
                # 如果尺寸发生变化，则进行裁剪
                if new_w != w or new_h != h:
                    # 这里选择从左上角裁剪 (0,0)，也可以选择中心裁剪
                    # crop box: (left, top, right, bottom)
                    img_cropped = img.crop((0, 0, new_w, new_h))
                    img_cropped.save(dst_path, quality=100) # png lossless, jpg high quality
                else:
                    # 尺寸本来就符合，直接复制保存（或者不做处理）
                    img.save(dst_path, quality=100)
                    
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 配置路径
    # 注意：这里使用了相对路径，请确保在正确的目录下运行
    source_path = "../Dataset/DIV2K/DIV2K_train_HR"
    target_path = "../Dataset/DIV2K/DIV2K_train_HR_test"
    
    # 检查源目录是否存在
    if not os.path.exists(source_path):
        print(f"错误：源目录不存在 -> {source_path}")
        # 尝试打印当前工作目录帮助调试
        print(f"当前工作目录: {os.getcwd()}")
    else:
        resize_and_save_images(source_path, target_path, base_num=16)
        print("所有图片处理完成！")