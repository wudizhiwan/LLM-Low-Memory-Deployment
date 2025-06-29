import torch
import os
import gc
import time
from pathlib import Path
from transformers import AutoTokenizer
from qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeConfig, Qwen3MoeRMSNorm, Qwen3MoeRotaryEmbedding
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

class Qwen3MoeLayerByLayerTest:
    """逐层加载并测试Qwen3 MOE模型的前向传播"""
    def __init__(self, model_path, weights_dir=None, device="cpu"):
        print(f"初始化测试，使用设备: {device}")
        print_memory_usage("初始化前")
        
        self.device = device
        self.model_path = model_path
        self.weights_dir = Path(weights_dir) if weights_dir else Path(model_path) / "layers"
        
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
        self.config = Qwen3MoeConfig.from_pretrained(model_path)
        print(f"模型配置: 隐藏层大小={self.config.hidden_size}, 层数={self.config.num_hidden_layers}, 注意力头数={self.config.num_attention_heads}")
        # 检查MOE特有参数
        if hasattr(self.config, "num_experts"):
            print(f"MOE配置: 专家数量={self.config.num_experts}, 每个token使用专家数量={self.config.num_experts_per_tok}")
        print_memory_usage("加载配置后")
        
        # 初始化旋转位置编码
        print("初始化旋转位置编码...")
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=self.config).to(self.device)
        print_memory_usage("初始化旋转位置编码后")
        
        # 预先打开词嵌入权重文件，但不加载到内存
        print("准备词嵌入权重...")
        self.embed_path = self.weights_dir / "embed_tokens.safetensors"
        
        # 预先打开最终层权重文件，但不加载到内存
        print("准备最终层权重...")
        self.final_layer_path = self.weights_dir / "final_layer.safetensors"
        
        # 初始化最终层归一化，但暂不加载权重
        with torch.no_grad():
            self.norm = Qwen3MoeRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(self.device)
            self.norm.eval()
            # 关闭梯度计算
            for param in self.norm.parameters():
                param.requires_grad_(False)
        
        print("初始化完成")
        print_memory_usage("初始化完成")
    
    def _load_embed_weights(self):
        """按需加载词嵌入权重"""
        print("加载词嵌入权重...")
        embed_weights = load_file(self.embed_path)
        embed_weight = embed_weights["model.embed_tokens.weight"].to(self.device, dtype=torch.bfloat16)
        print_memory_usage("加载词嵌入权重后")
        # 清理不再需要的变量
        del embed_weights
        clean_memory()
        return embed_weight
    
    def _load_norm_weights(self):
        """按需加载最终层归一化权重"""
        print("加载最终层归一化权重...")
        with safe_open(self.final_layer_path, framework="pt") as f:
            norm_weight = f.get_tensor("model.norm.weight").to(self.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            self.norm.weight.copy_(norm_weight)
        
        print_memory_usage("加载最终层归一化权重后")
        # 清理不再需要的变量
        del norm_weight
        clean_memory()
    
    def _load_lm_head_weights(self):
        """按需加载语言模型头权重"""
        print("加载语言模型头权重...")
        with safe_open(self.final_layer_path, framework="pt") as f:
            lm_head_weight = f.get_tensor("lm_head.weight").to(self.device, dtype=torch.bfloat16)
        
        print_memory_usage("加载语言模型头权重后")
        # 清理不再需要的变量后返回
        return lm_head_weight
    
    def _create_and_load_layer(self, layer_idx):
        """创建并加载指定层"""
        print(f"创建并加载第 {layer_idx} 层...")
        
        # 创建层
        with torch.no_grad():
            layer = Qwen3MoeDecoderLayer(self.config, layer_idx=layer_idx).to(self.device)
            layer.eval()
            
            # 关闭梯度计算
            for param in layer.parameters():
                param.requires_grad_(False)
        
        # 加载层权重
        layer_path = self.weights_dir / f"layer_{layer_idx}.safetensors"
        
        # 使用safe_open逐个加载权重，而不是一次性加载整个文件
        with safe_open(layer_path, framework="pt") as f:
            tensor_names = f.keys()
            prefix = f"model.layers.{layer_idx}."
            
            for name in tensor_names:
                if name.startswith(prefix):
                    # 移除"model.layers.N."前缀
                    param_name = name[len(prefix):]
                    
                    # 将参数名解析为嵌套属性访问
                    parts = param_name.split('.')
                    current = layer
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
                                        tensor = f.get_tensor(name).to(self.device, dtype=torch.bfloat16)
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
        
        print_memory_usage(f"加载第 {layer_idx} 层后")
        # 清理内存
        clean_memory()
        
        return layer
    
    def process_all_layers(self, prompt):
        """依次加载所有层并完成前向传播"""
        print(f"\n开始处理所有层，输入: \"{prompt}\"")
        print_memory_usage("开始处理")
        
        # 记录开始时间
        start_time = time.time()
        
        # 对输入进行编码
        print("对输入进行编码...")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        input_length = input_ids.shape[1]
        print(f"输入长度: {input_length} tokens")
        
        # 准备初始注意力掩码
        attention_mask = torch.ones((1, input_length), device=self.device)
        
        # 计算词嵌入
        print("计算词嵌入...")
        embed_weight = self._load_embed_weights()
        hidden_states = torch.nn.functional.embedding(input_ids, embed_weight)
        # 释放词嵌入权重
        del embed_weight
        clean_memory()
        
        # 准备位置ID
        position_ids = torch.arange(input_length, device=self.device).unsqueeze(0)
        
        # 计算旋转位置编码
        print("计算旋转位置编码...")
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # 准备因果掩码
        print("准备因果掩码...")
        dtype = hidden_states.dtype
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (input_length, input_length), 
            fill_value=min_dtype, 
            dtype=dtype, 
            device=self.device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # 依次加载并处理每一层
        print(f"开始依次处理{self.config.num_hidden_layers}个层...")
        router_logits_all = []  # 用于收集所有路由器的logits
        
        for layer_idx in range(self.config.num_hidden_layers):
            print(f"\n处理第 {layer_idx} 层...")
            
            # 创建并加载当前层
            current_layer = self._create_and_load_layer(layer_idx)
            
            # 测量加载完整层后的内存占用
            print_memory_usage(f"加载完整第 {layer_idx} 层后")
            
            # 执行当前层的前向传播
            print(f"执行第 {layer_idx} 层前向传播...")
            with torch.no_grad():
                layer_outputs = current_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False,
                    output_router_logits=True  # 请求MOE路由器的logits
                )
            
            # 更新隐藏状态
            hidden_states = layer_outputs[0]
            
            # 如果返回了路由器logits，收集它们
            if len(layer_outputs) > 1 and layer_outputs[-1] is not None:
                router_logits_all.append(layer_outputs[-1])
                print(f"获取到MOE路由器logits，形状: {layer_outputs[-1].shape}")
            
            # 确保hidden_states保持bfloat16类型
            if hidden_states.dtype != torch.bfloat16:
                hidden_states = hidden_states.to(torch.bfloat16)
            
            # 释放当前层内存
            del current_layer
            del layer_outputs
            clean_memory()
            
            # 每5层打印一次内存使用情况
            if layer_idx % 5 == 4:
                print_memory_usage(f"处理完第 {layer_idx} 层后")
        
        # 加载并应用最终层归一化
        print("\n加载并应用最终层归一化...")
        self._load_norm_weights()
        with torch.no_grad():
            hidden_states = self.norm(hidden_states)
            # 确保hidden_states保持bfloat16类型
            if hidden_states.dtype != torch.bfloat16:
                hidden_states = hidden_states.to(torch.bfloat16)
        
        # 应用语言模型头获取logits
        print("应用语言模型头获取logits...")
        lm_head_weight = self._load_lm_head_weights()
        
        # 打印数据类型信息以便调试
        print(f"hidden_states dtype: {hidden_states.dtype}, lm_head_weight dtype: {lm_head_weight.dtype}")
        
        logits = torch.nn.functional.linear(hidden_states, lm_head_weight)
        
        # 释放语言模型头权重
        del lm_head_weight
        clean_memory()
        
        # 获取预测的token
        print("获取预测的token...")
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        
        # 解码输出
        print("解码输出...")
        output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        predicted_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        
        # 计算总用时
        total_time = time.time() - start_time
        print(f"总用时: {total_time:.2f}秒")
        
        print(f"输入文本: {output_text}")
        print(f"预测的下一个token: {predicted_token}")
        print_memory_usage("处理完成")
        
        # 如果收集了路由器logits，进行一些分析
        if router_logits_all:
            print("\nMOE路由器分析:")
            for i, logits in enumerate(router_logits_all):
                if logits is not None:
                    # 计算平均激活的专家
                    routing_weights = torch.nn.functional.softmax(logits, dim=1)
                    top_experts = torch.topk(routing_weights, self.config.num_experts_per_tok, dim=1).indices
                    expert_counts = torch.bincount(top_experts.flatten(), minlength=self.config.num_experts)
                    expert_usage = expert_counts.float() / expert_counts.sum()
                    print(f"层 {i} 专家使用情况: {expert_usage.cpu().numpy()}")
        
        # 返回结果
        return {
            "input_text": output_text,
            "predicted_token": predicted_token,
            "hidden_states": hidden_states,
            "logits": logits,
            "total_time": total_time,
            "router_logits": router_logits_all if router_logits_all else None
        }
    
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
            # 释放词嵌入权重
            del embed_weight
            clean_memory()
            
            # 计算旋转位置编码
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            
            # 准备因果掩码
            input_length = hidden_states.shape[1]
            dtype = hidden_states.dtype
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (input_length, input_length), 
                fill_value=min_dtype, 
                dtype=dtype, 
                device=self.device
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # 依次加载并处理每一层
            for layer_idx in range(self.config.num_hidden_layers):
                # 创建并加载当前层
                current_layer = self._create_and_load_layer(layer_idx)
                
                # 测量加载完整层后的内存占用
                print_memory_usage(f"生成token {i+1} - 加载完整第 {layer_idx} 层后")
                
                # 执行当前层的前向传播
                with torch.no_grad():
                    layer_outputs = current_layer(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        use_cache=False,
                        output_attentions=False,
                        output_router_logits=False  # 生成时不需要路由器logits
                    )
                
                # 更新隐藏状态
                hidden_states = layer_outputs[0]
                
                # 确保hidden_states保持bfloat16类型
                if hidden_states.dtype != torch.bfloat16:
                    hidden_states = hidden_states.to(torch.bfloat16)
                
                # 释放当前层内存
                del current_layer
                del layer_outputs
                clean_memory()
            
            # 加载并应用最终层归一化
            self._load_norm_weights()
            with torch.no_grad():
                hidden_states = self.norm(hidden_states)
                # 确保hidden_states保持bfloat16类型
                if hidden_states.dtype != torch.bfloat16:
                    hidden_states = hidden_states.to(torch.bfloat16)
            
            # 应用语言模型头获取logits
            lm_head_weight = self._load_lm_head_weights()
            logits = torch.nn.functional.linear(hidden_states, lm_head_weight)
            # 释放语言模型头权重
            del lm_head_weight
            clean_memory()
            
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
            del hidden_states, logits, next_token_logits
            clean_memory()
        
        # 解码完整输出
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 计算总用时
        total_time = time.time() - start_time
        print(f"\n生成完成，总用时: {total_time:.2f}秒，平均每token: {total_time/max_new_tokens:.2f}秒")
        print(f"生成的完整文本: {output_text}")
        print_memory_usage("生成结束")
        
        return output_text

def main():
    # 设置模型路径
    model_path = "/home/szm/qwen3_a3b"  # 修改为MOE模型路径
    weights_dir = "moe_layers"  # 修改为MOE权重分割目录
    
    # 打印初始内存使用情况
    print("初始内存使用情况:")
    print_memory_usage("程序开始")
    
    try:
        # 初始化测试器 - 强制使用CPU
        tester = Qwen3MoeLayerByLayerTest(model_path, weights_dir=weights_dir, device="cpu")
        
        # 设置测试输入
        prompt = "介绍一下大语言模型"
        
        # 方法1: 处理所有层并获取下一个token预测
        print("\n===== 方法1: 处理所有层并获取下一个token预测 =====")
        result = tester.process_all_layers(prompt)
        
        # 方法2: 生成多个token形成文本
        print("\n===== 方法2: 生成多个token形成文本 =====")
        generated_text = tester.generate_text(prompt, max_new_tokens=20) # 减少token数量以加快测试
        
        print("\n测试完成!")
        print_memory_usage("程序结束")
        
        # 清理内存
        del tester
        clean_memory()
        
    except Exception as e:
        import traceback
        print(f"发生错误: {str(e)}")
        traceback.print_exc()
        print_memory_usage("错误发生后")

if __name__ == "__main__":
    main() 