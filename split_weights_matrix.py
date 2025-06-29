import torch
import os
import json
from pathlib import Path
from safetensors.torch import load_file, save_file

def print_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    print(f"当前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def split_weights(model_path, output_dir):
    print("开始分 割模型权重...")
    print_memory_usage()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型权重
    print(f"加载模型权重文件: {model_path}")
    weights = load_file(model_path)
    print(f"权重文件加载完成，包含 {len(weights)} 个张量")
    print_memory_usage()
    
    # 获取模型配置
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    num_layers = config.get('num_hidden_layers', 0)
    
    print(f"模型层数: {num_layers}")
    
    # 1. 分割词嵌入层
    print("处理词嵌入层...")
    embed_chunk = {k: v for k, v in weights.items() if 'embed_tokens' in k}
    if embed_chunk:
        output_path = os.path.join(output_dir, 'embed_tokens.safetensors')
        print(f"保存词嵌入层到 {output_path}")
        save_file(embed_chunk, output_path)
        print_memory_usage()
    
    # 2. 按矩阵级别分割每一层Transformer
    print("处理Transformer层...")
    for layer_idx in range(num_layers):
        print(f"处理第 {layer_idx + 1}/{num_layers} 层...")
        layer_prefix = f'model.layers.{layer_idx}.'
        
        # 2.1 分割自注意力部分
        attn_chunk = {k: v for k, v in weights.items() if k.startswith(layer_prefix) and 'self_attn' in k}
        if attn_chunk:
            output_path = os.path.join(output_dir, f'layer_{layer_idx}_attention.safetensors')
            print(f"保存第 {layer_idx} 层注意力部分到 {output_path}")
            save_file(attn_chunk, output_path)
            print_memory_usage()
        
        # 2.2 分割MLP部分 - gate
        mlp_gate_chunk = {k: v for k, v in weights.items() if k.startswith(layer_prefix) and 'mlp.gate_proj' in k}
        if mlp_gate_chunk:
            output_path = os.path.join(output_dir, f'layer_{layer_idx}_mlp_gate.safetensors')
            print(f"保存第 {layer_idx} 层MLP门控部分到 {output_path}")
            save_file(mlp_gate_chunk, output_path)
            print_memory_usage()
        
        # 2.3 分割MLP部分 - up
        mlp_up_chunk = {k: v for k, v in weights.items() if k.startswith(layer_prefix) and 'mlp.up_proj' in k}
        if mlp_up_chunk:
            output_path = os.path.join(output_dir, f'layer_{layer_idx}_mlp_up.safetensors')
            print(f"保存第 {layer_idx} 层MLP上投影部分到 {output_path}")
            save_file(mlp_up_chunk, output_path)
            print_memory_usage()
        
        # 2.4 分割MLP部分 - down
        mlp_down_chunk = {k: v for k, v in weights.items() if k.startswith(layer_prefix) and 'mlp.down_proj' in k}
        if mlp_down_chunk:
            output_path = os.path.join(output_dir, f'layer_{layer_idx}_mlp_down.safetensors')
            print(f"保存第 {layer_idx} 层MLP下投影部分到 {output_path}")
            save_file(mlp_down_chunk, output_path)
            print_memory_usage()
        
        # 2.5 分割层归一化部分
        norm_chunk = {k: v for k, v in weights.items() if k.startswith(layer_prefix) and 'input_layernorm' in k or k.startswith(layer_prefix) and 'post_attention_layernorm' in k}
        if norm_chunk:
            output_path = os.path.join(output_dir, f'layer_{layer_idx}_norm.safetensors')
            print(f"保存第 {layer_idx} 层归一化部分到 {output_path}")
            save_file(norm_chunk, output_path)
            print_memory_usage()
    
    # 3. 分割最终层（norm和lm_head）
    print("处理最终层...")
    final_chunk = {k: v for k, v in weights.items() if 'norm' in k and not any(f'layers.{i}' in k for i in range(num_layers)) or 'lm_head' in k}
    if final_chunk:
        output_path = os.path.join(output_dir, 'final_layer.safetensors')
        print(f"保存最终层到 {output_path}")
        save_file(final_chunk, output_path)
        print_memory_usage()
    
    print("权重分割完成！")
    print_memory_usage()

def main():
    # 设置路径
    model_path = "weight/model.safetensors"
    output_dir = "split_weights_matrix"
    
    # 分割权重
    split_weights(model_path, output_dir)

if __name__ == "__main__":
    main() 