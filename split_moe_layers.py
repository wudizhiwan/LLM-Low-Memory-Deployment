import torch
import os
import json
import glob
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def print_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    print(f"当前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def split_moe_weights(model_path, output_dir):
    print("开始 分割MOE模型权重...")
    print_memory_usage()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查模型路径是文件还是目录
    model_path = Path(model_path)
    if model_path.is_file():
        # 单一权重文件的情况
        weight_files = [model_path]
    else:
        # 目录中包含多个权重文件的情况
        print(f"检测到模型目录: {model_path}")
        # 查找所有的safetensors文件
        weight_files = list(model_path.glob("*.safetensors"))
        if not weight_files:
            # 如果没有直接的safetensors文件，尝试查找模型目录下可能的子目录
            weight_files = list(model_path.glob("**/*.safetensors"))
        
        if not weight_files:
            # 查找PyTorch的.bin文件作为备选
            weight_files = list(model_path.glob("*.bin"))
            if not weight_files:
                weight_files = list(model_path.glob("**/*.bin"))
        
        print(f"找到 {len(weight_files)} 个权重文件")
    
    # 获取模型配置
    config_path = model_path / "config.json"
    if not config_path.exists():
        # 尝试在模型目录下查找
        config_files = list(model_path.glob("**/config.json"))
        if config_files:
            config_path = config_files[0]
        else:
            raise FileNotFoundError(f"无法找到模型配置文件: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    num_layers = config.get('num_hidden_layers', 0)
    
    # 检查是否为MOE模型
    is_moe = False
    num_experts = 0
    experts_per_token = 0
    
    if 'num_local_experts' in config:
        is_moe = True
        num_experts = config.get('num_local_experts', 0)
        experts_per_token = config.get('num_experts_per_tok', 0)
        print(f"检测到MOE模型: {num_experts}个专家, 每个token使用{experts_per_token}个专家")
    
    print(f"模型层数: {num_layers}")
    
    # 创建一个合并所有权重的字典
    all_weights = {}
    
    # 逐个加载权重文件并合并
    for weight_file in tqdm(weight_files, desc="加载权重文件"):
        print(f"\n加载权重文件: {weight_file}")
        try:
            # 指定device="cpu"来避免GPU设备错误
            weights = load_file(str(weight_file), device="cpu")
            print(f"文件 {weight_file.name} 加载完成，包含 {len(weights)} 个张量")
            # 合并权重
            all_weights.update(weights)
            # 立即释放内存
            del weights
            import gc
            gc.collect()
            print_memory_usage()
        except Exception as e:
            print(f"加载文件 {weight_file} 时出错: {str(e)}")
    
    print(f"所有权重加载完成，共 {len(all_weights)} 个张量")
    print_memory_usage()
    
    # 1. 分割词嵌入层
    print("处理词嵌入层...")
    embed_chunk = {k: v for k, v in all_weights.items() if 'embed_tokens' in k}
    if embed_chunk:
        output_path = os.path.join(output_dir, 'embed_tokens.safetensors')
        print(f"保存词嵌入层到 {output_path}")
        save_file(embed_chunk, output_path)
        # 释放内存
        del embed_chunk
        gc.collect()
        print_memory_usage()
    
    # 2. 分割每一层Transformer (包括MOE专家)
    print("处理Transformer层...")
    for layer_idx in range(num_layers):
        print(f"处理第 {layer_idx + 1}/{num_layers} 层...")
        layer_chunk = {k: v for k, v in all_weights.items() if f'layers.{layer_idx}.' in k}
        if layer_chunk:
            output_path = os.path.join(output_dir, f'layer_{layer_idx}.safetensors')
            print(f"保存第 {layer_idx} 层到 {output_path}")
            save_file(layer_chunk, output_path)
            # 释放内存
            del layer_chunk
            gc.collect()
            print_memory_usage()
    
    # 3. 分割最终层（norm和lm_head）
    print("处理最终层...")
    # 更精确地匹配最终层，避免与层内部的norm混淆
    final_chunk = {k: v for k, v in all_weights.items() if ('model.norm' in k or 'lm_head' in k)}
    if final_chunk:
        output_path = os.path.join(output_dir, 'final_layer.safetensors')
        print(f"保存最终层到 {output_path}")
        save_file(final_chunk, output_path)
        # 释放内存
        del final_chunk
        gc.collect()
        print_memory_usage()
    
    # 4. 处理可能的其他权重（如果有）
    remaining_weights = {k: v for k, v in all_weights.items() 
                        if not any(x in k for x in ['embed_tokens', 'layers.', 'model.norm', 'lm_head'])}
    if remaining_weights:
        output_path = os.path.join(output_dir, 'other_weights.safetensors')
        print(f"保存其他权重到 {output_path}")
        save_file(remaining_weights, output_path)
        # 释放内存
        del remaining_weights
        gc.collect()
        print_memory_usage()
    
    # 释放所有权重的内存
    del all_weights
    gc.collect()
    
    print("MOE模型权重分割完成！")
    print_memory_usage()

def main():
    # 设置路径
    model_path = "/home/szm/qwen3_a3b"  # 请替换为您的MOE模型路径
    output_dir = "moe_layers"
    
    # 分割权重
    split_moe_weights(model_path, output_dir)

if __name__ == "__main__":
    main() 