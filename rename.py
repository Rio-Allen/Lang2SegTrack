import os
import sys
import shutil

def rename_mp4_files(root_dir, output_dir):
    """
    遍历目录，将每个子目录中的mp4文件重命名为三位数字格式并复制到新目录
    
    Args:
        root_dir: 根目录路径
        output_dir: 输出目录路径
    """
    # 检查源目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        return
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 遍历根目录下的所有子目录
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        
        # 只处理子目录
        if not os.path.isdir(subdir_path):
            continue
        
        print(f"\n处理目录: {subdir}")
        
        # 创建对应的输出子目录
        output_subdir = os.path.join(output_dir, subdir)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        # 获取所有mp4文件并排序
        mp4_files = [f for f in os.listdir(subdir_path) if f.endswith('.mp4')]
        mp4_files.sort()  # 按文件名排序
        
        # 复制并重命名文件
        for index, old_name in enumerate(mp4_files):
            old_path = os.path.join(subdir_path, old_name)
            new_name = f"{index:03d}.mp4"
            new_path = os.path.join(output_subdir, new_name)
            
            # 复制文件到新位置
            shutil.copy2(old_path, new_path)
            print(f"  {old_name} -> {new_name}")
        
        print(f"  完成: 共处理 {len(mp4_files)} 个文件")

if __name__ == "__main__":
    # 从命令行参数获取目录
    if len(sys.argv) >= 3:
        source_dir = sys.argv[1]
        target_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        source_dir = sys.argv[1]
        # 默认输出目录为源目录同级的 _renamed 目录
        target_dir = source_dir.rstrip('/') + '_renamed'
    else:
        source_dir = "/root/paddlejob/workspace/env_run/xuruihao/projects/FoodMonitor/Video_oneminute"
        target_dir = "/root/paddlejob/workspace/env_run/xuruihao/projects/Video_oneminute"
        if not target_dir:
            target_dir = source_dir.rstrip('/') + '_renamed'
    
    print(f"源目录: {source_dir}")
    print(f"输出目录: {target_dir}")
    
    rename_mp4_files(source_dir, target_dir)
    print("\n重命名完成！")