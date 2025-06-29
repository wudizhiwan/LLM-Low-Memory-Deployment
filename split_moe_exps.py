import torch
import os
import json
import gc
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def print_memory_usage():
    """ 打印当前内存使用情况"""
    import psutil
    process = psutil.Process(os.getpid())
    print(f"当前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def split_moe_experts(model_path, output_dir):
    """将MO   E模型权重按更细粒度分割，特别是将每层的专家分别存储"""
    print("开始细粒度分割MOE模型权重...")
    print_memory_usage()
    
    # 创建输出主目录
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
    
    # 复制配置文件到输出目录
    import shutil
    output_config_path = Path(output_dir) / "config.json"
    shutil.copy(config_path, output_config_path)
    print(f"配置文件已复制到 {output_config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    num_layers = config.get('num_hidden_layers', 0)
    
    # 检查是否为MOE模型
    is_moe = False
    num_experts = 0
    experts_per_token = 0
    
    # 检查多种可能的MOE参数命名方式
    if 'num_local_experts' in config:
        is_moe = True
        num_experts = config.get('num_local_experts', 0)
        experts_per_token = config.get('num_experts_per_tok', 0)
    elif 'num_experts' in config:
        is_moe = True
        num_experts = config.get('num_experts', 0)
        experts_per_token = config.get('num_experts_per_tok', 0)
    
    if is_moe and num_experts > 0:
        print(f"检测到MOE模型: {num_experts}个专家, 每个token使用{experts_per_token}个专家")
    else:
        # 检查模型类型名称
        model_type = config.get('model_type', '').lower()
        architectures = config.get('architectures', [])
        
        if 'moe' in model_type or any('moe' in arch.lower() for arch in architectures):
            is_moe = True
            # 尝试从配置中推断专家数量
            num_experts = config.get('num_experts', 8)  # 默认值8
            experts_per_token = config.get('num_experts_per_tok', 2)  # 默认值2
            print(f"通过模型类型检测到MOE模型: {num_experts}个专家, 每个token使用{experts_per_token}个专家")
        else:
            raise ValueError("这不是一个MOE模型，请检查配置文件或手动设置num_experts参数")
    
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
    
    # 2. 分割每一层Transformer，细分为注意力层、路由器和各个专家
    print("处理Transformer层...")
    for layer_idx in range(num_layers):
        print(f"处理第 {layer_idx + 1}/{num_layers} 层...")
        
        # 为当前层创建目录
        layer_dir = os.path.join(output_dir, f'layer_{layer_idx}')
        os.makedirs(layer_dir, exist_ok=True)
        
        # 获取该层的所有权重
        layer_weights = {k: v for k, v in all_weights.items() if f'layers.{layer_idx}.' in k}
        
        # 2.1 提取注意力层权重
        attention_weights = {}
        for k, v in layer_weights.items():
            if 'self_attn' in k:
                attention_weights[k] = v
        
        if attention_weights:
            attn_output_path = os.path.join(layer_dir, 'attention.safetensors')
            print(f"保存第 {layer_idx} 层注意力权重到 {attn_output_path}")
            save_file(attention_weights, attn_output_path)
            # 释放内存
            del attention_weights
            gc.collect()
        
        # 2.2 提取层归一化权重
        norm_weights = {}
        for k, v in layer_weights.items():
            if 'input_layernorm' in k or 'post_attention_layernorm' in k:
                norm_weights[k] = v
        
        if norm_weights:
            norm_output_path = os.path.join(layer_dir, 'norm.safetensors')
            print(f"保存第 {layer_idx} 层归一化权重到 {norm_output_path}")
            save_file(norm_weights, norm_output_path)
            # 释放内存
            del norm_weights
            gc.collect()
        
        # 2.3 提取路由器权重
        router_weights = {}
        for k, v in layer_weights.items():
            if 'mlp.gate' in k and not any(f'experts.{i}' in k for i in range(num_experts)):
                router_weights[k] = v
        
        if router_weights:
            router_output_path = os.path.join(layer_dir, 'router.safetensors')
            print(f"保存第 {layer_idx} 层路由器权重到 {router_output_path}")
            save_file(router_weights, router_output_path)
            # 释放内存
            del router_weights
            gc.collect()
        
        # 2.4 创建专家目录并分别保存每个专家的权重
        experts_dir = os.path.join(layer_dir, 'experts')
        os.makedirs(experts_dir, exist_ok=True)
        
        for expert_idx in range(num_experts):
            expert_weights = {}
            for k, v in layer_weights.items():
                if f'mlp.experts.{expert_idx}' in k:
                    expert_weights[k] = v
            
            if expert_weights:
                expert_output_path = os.path.join(experts_dir, f'expert_{expert_idx}.safetensors')
                print(f"保存第 {layer_idx} 层专家 {expert_idx} 权重到 {expert_output_path}")
                save_file(expert_weights, expert_output_path)
                # 释放内存
                del expert_weights
                gc.collect()
        
        # 释放该层的所有权重内存
        del layer_weights
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
    
    print("MOE模型专家权重细粒度分割完成！")
    print("目录结构:")
    print(f"{output_dir}/")
    print(f"├── embed_tokens.safetensors")
    print(f"├── final_layer.safetensors")
    print(f"├── config.json")
    for i in range(min(3, num_layers)):
        print(f"├── layer_{i}/")
        print(f"│   ├── attention.safetensors")
        print(f"│   ├── norm.safetensors")
        print(f"│   ├── router.safetensors")
        print(f"│   └── experts/")
        for j in range(min(3, num_experts)):
            print(f"│       ├── expert_{j}.safetensors")
        if num_experts > 3:
            print(f"│       └── ...")
    if num_layers > 3:
        print(f"└── ...")
    
    print_memory_usage()

def main():
    # 设置路径
    model_path = "/home/szm/qwen3_a3b"  # 请替换为您的MOE模型路径
    output_dir = "moe_experts"  # 更细粒度分割的输出目录
    
    # 分割权重
    split_moe_experts(model_path, output_dir)

if __name__ == "__main__":
    main() 