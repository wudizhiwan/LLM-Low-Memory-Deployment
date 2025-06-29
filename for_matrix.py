import torch
import os
import gc
import time
from pathlib import Path
from transformers import AutoTokenizer
from qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Config, Qwen3RMSNorm, Qwen3RotaryEmbedding
from safetensors.torch import load_file, safe_open
import psutil

def print_memory_usage(label=""):
    """打印当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"[{label}] 当前内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # 打印系统内存信息
    virtual_memory = psutil.virtual_memory()
    print(f"系统内存: 总计={virtual_memory.total / 1024 / 1024 / 1024:.2f}GB, 可用={virtual_memory.available / 1024 / 1024 / 1024:.2f}GB, 使用率={virtual_memory.percent}%")

def clean_memory():
    """清理内存"""
    gc.collect()

# 添加内存池管理，减少频繁分配/释放内存的开销
class TensorCache:
    """张量缓存，用于减少重复加载和内存碎片化"""
    def __init__(self, max_size=5):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key):
        """获取缓存的张量"""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key, tensor):
        """存入张量到缓存"""
        # 如果缓存已满，移除最少访问的项
        if len(self.cache) >= self.max_size and key not in self.cache:
            least_used_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_used_key]
            del self.access_count[least_used_key]
        
        self.cache[key] = tensor
        self.access_count[key] = 1
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
        clean_memory()

class MLPComponent:
    """MLP组件，用于后续可能的MoE替换"""
    def __init__(self, config, layer_idx, device="cpu", dtype=torch.float16, weights_dir=None, tensor_cache=None):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.dtype = dtype
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.tensor_cache = tensor_cache
        
        # 初始化MLP权重
        self.gate_proj_weight = None
        self.up_proj_weight = None
        self.down_proj_weight = None
        
        # 加载权重
        if self.weights_dir:
            self.load_weights()
    
    def load_weights(self):
        """加载MLP权重"""
        # 加载gate投影权重
        gate_key = f"mlp_gate_{self.layer_idx}"
        if self.tensor_cache and (cached_tensor := self.tensor_cache.get(gate_key)) is not None:
            self.gate_proj_weight = cached_tensor
        else:
            gate_path = self.weights_dir / f"layer_{self.layer_idx}_mlp_gate.safetensors"
            if gate_path.exists():
                with safe_open(gate_path, framework="pt") as f:
                    key = f"model.layers.{self.layer_idx}.mlp.gate_proj.weight"
                    self.gate_proj_weight = f.get_tensor(key).to(self.device, dtype=self.dtype)
                    if self.tensor_cache:
                        self.tensor_cache.put(gate_key, self.gate_proj_weight)
        
        # 加载up投影权重
        up_key = f"mlp_up_{self.layer_idx}"
        if self.tensor_cache and (cached_tensor := self.tensor_cache.get(up_key)) is not None:
            self.up_proj_weight = cached_tensor
        else:
            up_path = self.weights_dir / f"layer_{self.layer_idx}_mlp_up.safetensors"
            if up_path.exists():
                with safe_open(up_path, framework="pt") as f:
                    key = f"model.layers.{self.layer_idx}.mlp.up_proj.weight"
                    self.up_proj_weight = f.get_tensor(key).to(self.device, dtype=self.dtype)
                    if self.tensor_cache:
                        self.tensor_cache.put(up_key, self.up_proj_weight)
        
        # 加载down投影权重
        down_key = f"mlp_down_{self.layer_idx}"
        if self.tensor_cache and (cached_tensor := self.tensor_cache.get(down_key)) is not None:
            self.down_proj_weight = cached_tensor
        else:
            down_path = self.weights_dir / f"layer_{self.layer_idx}_mlp_down.safetensors"
            if down_path.exists():
                with safe_open(down_path, framework="pt") as f:
                    key = f"model.layers.{self.layer_idx}.mlp.down_proj.weight"
                    self.down_proj_weight = f.get_tensor(key).to(self.device, dtype=self.dtype)
                    if self.tensor_cache:
                        self.tensor_cache.put(down_key, self.down_proj_weight)
    
    def forward(self, x):
        """前向传播"""
        if self.gate_proj_weight is None or self.up_proj_weight is None or self.down_proj_weight is None:
            raise ValueError("MLP权重未加载")
        
        # 确保输入数据类型与权重一致，避免不必要的类型转换
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        
        # 计算gate和up投影，使用原地操作减少内存使用
        gate_output = torch.nn.functional.linear(x, self.gate_proj_weight)
        up_output = torch.nn.functional.linear(x, self.up_proj_weight)
        
        # SwiGLU激活函数
        intermediate = torch.nn.functional.silu(gate_output) * up_output
        
        # 释放不再需要的中间结果
        del gate_output, up_output
        
        # 下投影
        output = torch.nn.functional.linear(intermediate, self.down_proj_weight)
        
        # 释放中间结果
        del intermediate
        
        return output
    
    def cleanup(self):
        """清理权重"""
        # 不实际删除权重，因为它们可能在缓存中被重用
        self.gate_proj_weight = None
        self.up_proj_weight = None
        self.down_proj_weight = None

class MatrixLevelInference:
    """矩阵级别加载并推理Qwen3模型"""
    def __init__(self, model_path, weights_dir=None, device="cpu", dtype=torch.float16):
        print(f"初始化矩阵级别推理，使用设备: {device}, 数据类型: {dtype}")
        print_memory_usage("初始化前")
        
        self.device = "cpu"  # 强制使用CPU
        self.dtype = dtype
        self.model_path = model_path
        self.weights_dir = Path(weights_dir) if weights_dir else Path(model_path) / "layers"
        
        # 创建张量缓存
        self.tensor_cache = TensorCache(max_size=10)
        
        # 确保权重目录存在
        if not self.weights_dir.exists():
            raise ValueError(f"权重目录不存在: {self.weights_dir}")
        
        # 加载分词器
        print("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"分词器类型: {type(self.tokenizer)}")
        print_memory_usage("加载分词器后")
        
        # 加载配置
        print("加载配置...")
        self.config = Qwen3Config.from_pretrained(model_path)
        print(f"模型配置: 隐藏层大小={self.config.hidden_size}, 层数={self.config.num_hidden_layers}, 注意力头数={self.config.num_attention_heads}")
        print_memory_usage("加载配置后")
        
        # 初始化旋转位置编码
        print("初始化旋转位置编码...")
        self.rotary_emb = Qwen3RotaryEmbedding(config=self.config).to(self.device)
        print_memory_usage("初始化旋转位置编码后")
        
        # 初始化最终层归一化，但暂不加载权重
        with torch.no_grad():
            self.norm = Qwen3RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(self.device)
            self.norm.eval()
            # 关闭梯度计算并确保数据类型一致
            for param in self.norm.parameters():
                param.requires_grad_(False)
                param.data = param.data.to(self.dtype)
        
        print("初始化完成")
        print_memory_usage("初始化完成")
    
    def _load_embed_weights(self):
        """按需加载词嵌入权重"""
        embed_key = "embed_tokens"
        if cached_tensor := self.tensor_cache.get(embed_key):
            return cached_tensor
            
        print("加载词嵌入权重...")
        embed_path = self.weights_dir / "embed_tokens.safetensors"
        embed_weights = load_file(embed_path)
        embed_weight = embed_weights["model.embed_tokens.weight"].to(self.device, dtype=self.dtype)
        print_memory_usage("加载词嵌入权重后")
        # 清理不再需要的变量
        del embed_weights
        
        # 缓存权重
        self.tensor_cache.put(embed_key, embed_weight)
        return embed_weight
    
    def _load_norm_weights(self):
        """按需加载最终层归一化权重"""
        norm_key = "final_norm"
        if self.tensor_cache.get(norm_key) is not None:
            return
            
        print("加载最终层归一化权重...")
        final_layer_path = self.weights_dir / "final_layer.safetensors"
        with safe_open(final_layer_path, framework="pt") as f:
            norm_weight = f.get_tensor("model.norm.weight").to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            self.norm.weight.copy_(norm_weight)
        
        # 缓存权重状态
        self.tensor_cache.put(norm_key, True)
        
        print_memory_usage("加载最终层归一化权重后")
        # 清理不再需要的变量
        del norm_weight
    
    def _load_lm_head_weights(self):
        """按需加载语言模型头权重"""
        lm_head_key = "lm_head"
        if cached_tensor := self.tensor_cache.get(lm_head_key):
            return cached_tensor
            
        print("加载语言模型头权重...")
        final_layer_path = self.weights_dir / "final_layer.safetensors"
        with safe_open(final_layer_path, framework="pt") as f:
            lm_head_weight = f.get_tensor("lm_head.weight").to(self.device, dtype=self.dtype)
        
        # 缓存权重
        self.tensor_cache.put(lm_head_key, lm_head_weight)
        
        print_memory_usage("加载语言模型头权重后")
        return lm_head_weight
    
    def _load_layer_attention_weights(self, layer_idx, layer):
        """加载层的注意力权重"""
        attn_key = f"attn_{layer_idx}"
        if self.tensor_cache.get(attn_key) is not None:
            return
            
        print(f"加载第 {layer_idx} 层注意力权重...")
        attn_path = self.weights_dir / f"layer_{layer_idx}_attention.safetensors"
        
        with safe_open(attn_path, framework="pt") as f:
            tensor_names = f.keys()
            prefix = f"model.layers.{layer_idx}.self_attn."
            
            for name in tensor_names:
                if name.startswith(prefix):
                    # 移除前缀
                    param_name = name[len(prefix):]
                    
                    # 将参数名解析为嵌套属性访问
                    parts = param_name.split('.')
                    current = layer.self_attn
                    found = True
                    
                    # 遍历属性路径
                    for j, part in enumerate(parts):
                        if hasattr(current, part):
                            if j == len(parts) - 1:  # 最后一个部分是参数名
                                # 获取参数并复制权重
                                param = getattr(current, part)
                                if isinstance(param, torch.nn.Parameter):
                                    with torch.no_grad():
                                        # 直接从文件加载到设备和正确的数据类型
                                        tensor = f.get_tensor(name).to(self.device, dtype=self.dtype)
                                        param.copy_(tensor)
                                        # 立即删除临时张量
                                        del tensor
                                else:
                                    print(f"警告：{param_name} 不是参数")
                                    found = False
                            else:
                                current = getattr(current, part)
                        else:
                            print(f"警告：找不到属性 {part} 在 {'.'.join(parts[:j])}")
                            found = False
                            break
                    
                    if not found:
                        print(f"警告：在模型中找不到参数 {param_name}")
        
        # 标记已加载
        self.tensor_cache.put(attn_key, True)
        
        print_memory_usage(f"加载第 {layer_idx} 层注意力权重后")
    
    def _load_layer_norm_weights(self, layer_idx, layer):
        """加载层的归一化权重"""
        norm_key = f"norm_{layer_idx}"
        if self.tensor_cache.get(norm_key) is not None:
            return
            
        print(f"加载第 {layer_idx} 层归一化权重...")
        norm_path = self.weights_dir / f"layer_{layer_idx}_norm.safetensors"
        
        with safe_open(norm_path, framework="pt") as f:
            tensor_names = f.keys()
            prefix = f"model.layers.{layer_idx}."
            
            # 加载input_layernorm
            input_norm_name = f"{prefix}input_layernorm.weight"
            if input_norm_name in tensor_names:
                with torch.no_grad():
                    tensor = f.get_tensor(input_norm_name).to(self.device, dtype=self.dtype)
                    layer.input_layernorm.weight.copy_(tensor)
                    del tensor
            
            # 加载post_attention_layernorm
            post_norm_name = f"{prefix}post_attention_layernorm.weight"
            if post_norm_name in tensor_names:
                with torch.no_grad():
                    tensor = f.get_tensor(post_norm_name).to(self.device, dtype=self.dtype)
                    layer.post_attention_layernorm.weight.copy_(tensor)
                    del tensor
        
        # 标记已加载
        self.tensor_cache.put(norm_key, True)
        
        print_memory_usage(f"加载第 {layer_idx} 层归一化权重后")
    
    def _create_layer(self, layer_idx):
        """创建指定层"""
        layer_key = f"layer_{layer_idx}"
        if cached_layer := self.tensor_cache.get(layer_key):
            return cached_layer
            
        print(f"创建第 {layer_idx} 层...")
        
        # 创建层
        with torch.no_grad():
            layer = Qwen3DecoderLayer(self.config, layer_idx=layer_idx).to(self.device)
            layer.eval()
            
            # 关闭梯度计算并确保数据类型一致
            for param in layer.parameters():
                param.requires_grad_(False)
                param.data = param.data.to(self.dtype)
        
        # 缓存层
        self.tensor_cache.put(layer_key, layer)
        return layer
    
    def process_layer(self, layer_idx, hidden_states, attention_mask, position_ids, position_embeddings):
        """处理单层，分别加载注意力和MLP部分"""
        print(f"\n处理第 {layer_idx} 层...")
        
        # 确保输入数据类型正确
        if hidden_states.dtype != self.dtype:
            hidden_states = hidden_states.to(self.dtype)
        
        if position_embeddings is not None and position_embeddings[0].dtype != self.dtype:
            position_embeddings = tuple(pe.to(self.dtype) for pe in position_embeddings)
        
        # 创建层
        layer = self._create_layer(layer_idx)
        
        # 加载层归一化权重
        self._load_layer_norm_weights(layer_idx, layer)
        
        # 加载层注意力权重
        self._load_layer_attention_weights(layer_idx, layer)
        
        # 执行层归一化和注意力部分
        with torch.no_grad():
            # 应用输入层归一化
            norm_x = layer.input_layernorm(hidden_states)
            
            # 确保norm_x数据类型正确
            if norm_x.dtype != self.dtype:
                norm_x = norm_x.to(self.dtype)
            
            # 应用自注意力
            attn_outputs = layer.self_attn(
                norm_x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
            
            # 获取注意力输出并添加残差连接
            attn_output = attn_outputs[0]
            if attn_output.dtype != self.dtype:
                attn_output = attn_output.to(self.dtype)
            
            hidden_states = hidden_states + attn_output
            
            # 释放不再需要的变量
            del attn_outputs, attn_output, norm_x
            
            # 应用后注意力层归一化
            post_norm_x = layer.post_attention_layernorm(hidden_states)
            
            # 确保post_norm_x数据类型正确
            if post_norm_x.dtype != self.dtype:
                post_norm_x = post_norm_x.to(self.dtype)
        
        # 创建并加载MLP组件
        mlp = MLPComponent(
            config=self.config,
            layer_idx=layer_idx,
            device=self.device,
            dtype=self.dtype,
            weights_dir=self.weights_dir,
            tensor_cache=self.tensor_cache
        )
        
        # 应用MLP
        with torch.no_grad():
            # 这里可以替换为MoE实现
            mlp_output = mlp.forward(post_norm_x)
            
            # 确保mlp_output数据类型正确
            if mlp_output.dtype != self.dtype:
                mlp_output = mlp_output.to(self.dtype)
            
            # 添加残差连接
            hidden_states = hidden_states + mlp_output
            
            # 释放不再需要的变量
            del post_norm_x, mlp_output
        
        # 清理MLP组件
        mlp.cleanup()
        
        return hidden_states
    
    def generate_text(self, prompt, max_new_tokens=20):
        """生成文本"""
        print(f"\n开始生成文本，输入: \"{prompt}\"")
        print_memory_usage("生成开始")
        
        # 记录开始时间
        start_time = time.time()
        
        # 对输入进行编码
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        generated_ids = input_ids.clone()
        
        # 生成新token
        for i in range(max_new_tokens):
            token_start_time = time.time()
            print(f"\n生成第 {i+1}/{max_new_tokens} 个token...")
            
            # 准备位置ID
            position_ids = torch.arange(generated_ids.shape[1], device=self.device).unsqueeze(0)
            
            # 计算词嵌入
            embed_weight = self._load_embed_weights()
            hidden_states = torch.nn.functional.embedding(generated_ids, embed_weight)
            
            # 确保hidden_states数据类型正确
            if hidden_states.dtype != self.dtype:
                hidden_states = hidden_states.to(self.dtype)
            
            # 计算旋转位置编码
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            
            # 确保position_embeddings数据类型正确
            if position_embeddings is not None and position_embeddings[0].dtype != self.dtype:
                position_embeddings = tuple(pe.to(self.dtype) for pe in position_embeddings)
            
            # 准备因果掩码
            input_length = hidden_states.shape[1]
            dtype = self.dtype
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (input_length, input_length), 
                fill_value=min_dtype, 
                dtype=dtype, 
                device=self.device
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # 依次处理每一层
            for layer_idx in range(self.config.num_hidden_layers):
                hidden_states = self.process_layer(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )
                
                # 确保hidden_states保持正确的数据类型
                if hidden_states.dtype != self.dtype:
                    hidden_states = hidden_states.to(self.dtype)
            
            # 加载并应用最终层归一化
            self._load_norm_weights()
            with torch.no_grad():
                hidden_states = self.norm(hidden_states)
                # 确保hidden_states保持正确的数据类型
                if hidden_states.dtype != self.dtype:
                    hidden_states = hidden_states.to(self.dtype)
            
            # 应用语言模型头获取logits
            lm_head_weight = self._load_lm_head_weights()
            logits = torch.nn.functional.linear(hidden_states, lm_head_weight)
            
            # 获取预测的token
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # 添加到生成的序列中
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            # 解码当前token
            current_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            token_time = time.time() - token_start_time
            print(f"生成的token: {current_token} (用时: {token_time:.2f}秒)")
            
            # 每5个token打印一次内存使用情况
            if i % 5 == 4 or i == 0:
                print_memory_usage(f"生成第 {i+1} 个token后")
            
            # 检查是否生成了EOS token
            if next_token_id.item() == self.tokenizer.eos_token_id:
                print("遇到EOS token，停止生成")
                break
            
            # 释放不再需要的变量
            del logits, next_token_logits
        
        # 解码完整输出
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 计算总用时
        total_time = time.time() - start_time
        print(f"\n生成完成，总用时: {total_time:.2f}秒，平均每token: {total_time/max_new_tokens:.2f}秒")
        print(f"生成的完整文本: {output_text}")
        print_memory_usage("生成结束")
        
        # 清理缓存
        self.tensor_cache.clear()
        
        return output_text

def main():
    # 设置模型路径
    model_path = "weight"
    weights_dir = "split_weights_matrix"
    
    # 打印初始内存使用情况
    print("初始内存使用情况:")
    print_memory_usage("程序开始")
    
    try:
        # 初始化推理器 - 强制使用CPU
        device = "cpu"
        # 使用float16而不是float32
        dtype = torch.float16
        inferencer = MatrixLevelInference(model_path, weights_dir=weights_dir, device=device, dtype=dtype)
        
        # 设置测试输入
        prompt = "介绍一下大语言模型"
        
        # 生成文本
        print("\n===== 开始生成文本 =====")
        generated_text = inferencer.generate_text(prompt, max_new_tokens=20)  # 可以根据需要调整token数量
        
        print("\n测试完成!")
        print_memory_usage("程序结束")
        
        # 清理内存
        del inferencer
        clean_memory()
        
    except Exception as e:
        import traceback
        print(f"发生错误: {str(e)}")
        traceback.print_exc()
        print_memory_usage("错误发生后")

if __name__ == "__main__":
    main() 