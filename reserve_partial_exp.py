import torch
import os
import gc
import time
from pathlib import Path
from transformers import AutoTokenizer
from qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeConfig, Qwen3MoeRMSNorm, Qwen3MoeRotaryEmbedding, Qwen3MoeSparseMoeBlock
from safetensors.torch import load_file, safe_open
import psutil
import numpy as np

def print_memory_usage(label=""):
    """ 打印当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"[{label}] 当前内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # 打印系统内存信息
    virtual_memory = psutil.virtual_memory()
    print(f"系统内存: 总计={virtual_memory.total / 1024 / 1024 / 1024:.2f}GB, 可用={virtual_memory.available / 1024 / 1024 / 1024:.2f}GB, 使用率={virtual_memory.percent}%")

def clean_memory():
    """   清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class SelectiveExpertsMoeInference:
    """按需加载专家的MOE模型推理类"""
    def __init__(self, model_path, experts_dir=None, device="cpu"):
        print(f"初始化推理器，使用设备: {device}")
        print_memory_usage("初始化前")
        
        self.device = device
        self.model_path = model_path
        self.experts_dir = Path(experts_dir) if experts_dir else Path(model_path) / "moe_experts"
        
        # 确保权重目录存在
        if not self.experts_dir.exists():
            raise ValueError(f"专家权重目录不存在: {self.experts_dir}")
        
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
        self.embed_path = self.experts_dir / "embed_tokens.safetensors"
        
        # 预先打开最终层权重文件，但不加载到内存
        print("准备最终层权重...")
        self.final_layer_path = self.experts_dir / "final_layer.safetensors"
        
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
    
    def _create_empty_layer(self, layer_idx):
        """创建一个空的层，只包含基本结构但不加载权重"""
        print(f"创建空层 {layer_idx}...")
        with torch.no_grad():
            layer = Qwen3MoeDecoderLayer(self.config, layer_idx=layer_idx).to(self.device)
            layer.eval()
            
            # 关闭梯度计算
            for param in layer.parameters():
                param.requires_grad_(False)
        
        return layer
    
    def _load_attention_and_norm(self, layer, layer_idx):
        """加载层的注意力和归一化权重"""
        layer_dir = self.experts_dir / f"layer_{layer_idx}"
        
        # 加载注意力权重
        attn_path = layer_dir / "attention.safetensors"
        if attn_path.exists():
            print(f"加载第 {layer_idx} 层注意力权重...")
            with safe_open(attn_path, framework="pt") as f:
                tensor_names = f.keys()
                
                for name in tensor_names:
                    if name.startswith(f"model.layers.{layer_idx}."):
                        # 移除"model.layers.N."前缀
                        param_name = name[len(f"model.layers.{layer_idx}."):]
                        
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
                                        found = False
                                else:
                                    current = getattr(current, part)
                            else:
                                found = False
                                break
        
        # 加载归一化权重
        norm_path = layer_dir / "norm.safetensors"
        if norm_path.exists():
            print(f"加载第 {layer_idx} 层归一化权重...")
            with safe_open(norm_path, framework="pt") as f:
                tensor_names = f.keys()
                
                for name in tensor_names:
                    if name.startswith(f"model.layers.{layer_idx}."):
                        # 移除"model.layers.N."前缀
                        param_name = name[len(f"model.layers.{layer_idx}."):]
                        
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
                                        found = False
                                else:
                                    current = getattr(current, part)
                            else:
                                found = False
                                break
    
    def _load_router_weights(self, layer, layer_idx):
        """加载层的路由器权重"""
        router_path = self.experts_dir / f"layer_{layer_idx}" / "router.safetensors"
        if router_path.exists():
            print(f"加载第 {layer_idx} 层路由器权重...")
            with safe_open(router_path, framework="pt") as f:
                tensor_names = f.keys()
                
                for name in tensor_names:
                    if name.startswith(f"model.layers.{layer_idx}."):
                        # 移除"model.layers.N."前缀
                        param_name = name[len(f"model.layers.{layer_idx}."):]
                        
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
                                        found = False
                                else:
                                    current = getattr(current, part)
                            else:
                                found = False
                                break
    
    def _load_expert_weights(self, layer, layer_idx, expert_idx):
        """加载特定专家的权重"""
        expert_path = self.experts_dir / f"layer_{layer_idx}" / "experts" / f"expert_{expert_idx}.safetensors"
        if expert_path.exists():
            print(f"加载第 {layer_idx} 层专家 {expert_idx} 权重...")
            with safe_open(expert_path, framework="pt") as f:
                tensor_names = f.keys()
                
                for name in tensor_names:
                    if name.startswith(f"model.layers.{layer_idx}."):
                        # 移除"model.layers.N."前缀
                        param_name = name[len(f"model.layers.{layer_idx}."):]
                        
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
                                        found = False
                                else:
                                    current = getattr(current, part)
                            else:
                                found = False
                                break
    
    def _run_router_get_experts(self, layer, hidden_states):
        """运行路由器并获取选择的专家索引"""
        # 确保mlp是SparseMoeBlock类型
        if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            raise TypeError(f"该层的MLP不是SparseMoeBlock类型，而是 {type(layer.mlp)}")
        
        # 直接计算路由器logits，而不是完整执行前向传播
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, hidden_dim)
        router_logits = layer.mlp.gate(hidden_states_2d)
        
        # 计算路由权重并选择top-k专家
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        _, selected_experts = torch.topk(routing_weights, self.config.num_experts_per_tok, dim=-1)
        
        # 获取唯一的专家索引
        unique_experts = torch.unique(selected_experts).cpu().numpy()
        
        return unique_experts, selected_experts, routing_weights, router_logits
    
    def process_all_layers(self, prompt):
        """使用选择性专家加载方式处理所有层"""
        print(f"\n开始处理所有层，使用选择性专家加载方式，输入: \"{prompt}\"")
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
        
        # 收集各层专家分配信息的字典
        experts_stats = {}
        
        # 依次处理每一层
        print(f"开始依次处理{self.config.num_hidden_layers}个层...")
        
        for layer_idx in range(self.config.num_hidden_layers):
            print(f"\n处理第 {layer_idx} 层...")
            
            # 第一阶段：创建空层，加载注意力层、归一化层和路由器权重
            print(f"第一阶段: 加载第 {layer_idx} 层基础结构和路由器...")
            layer = self._create_empty_layer(layer_idx)
            self._load_attention_and_norm(layer, layer_idx)
            self._load_router_weights(layer, layer_idx)
            
            # 运行注意力机制的前向传播
            print(f"运行第 {layer_idx} 层注意力机制...")
            with torch.no_grad():
                # 执行自注意力部分的计算
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                
                # 应用自注意力
                hidden_states = layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]
                
                hidden_states = residual + hidden_states
            
            # 第二阶段：确定需要加载哪些专家
            print(f"第二阶段: 确定第 {layer_idx} 层需要的专家...")
            # 运行路由器获取专家选择
            residual = hidden_states
            hidden_states_for_router = layer.post_attention_layernorm(hidden_states)
            
            unique_experts, selected_experts, routing_weights, router_logits = self._run_router_get_experts(layer, hidden_states_for_router)
            
            # 记录专家使用情况
            experts_stats[f"layer_{layer_idx}"] = {
                "unique_experts": unique_experts.tolist(),
                "selected_experts_count": len(unique_experts),
                "total_experts": self.config.num_experts,
                "memory_saving_ratio": 1.0 - (len(unique_experts) / self.config.num_experts)
            }
            
            print(f"第 {layer_idx} 层需要加载 {len(unique_experts)}/{self.config.num_experts} 个专家 (节省了 {experts_stats[f'layer_{layer_idx}']['memory_saving_ratio']*100:.2f}% 的专家内存)")
            print(f"将加载的专家索引: {unique_experts}")
            
            # 第三阶段：只加载需要的专家权重
            print(f"第三阶段: 加载第 {layer_idx} 层选定的专家权重...")
            for expert_idx in unique_experts:
                self._load_expert_weights(layer, layer_idx, expert_idx)
            
            # 第四阶段：执行MLP (MOE) 的前向传播
            print(f"第四阶段: 执行第 {layer_idx} 层MOE前向传播...")
            with torch.no_grad():
                # 执行MLP (MOE) 部分
                # 只传递hidden_states_for_router，不传递router_logits
                if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                    moe_output, _ = layer.mlp(hidden_states_for_router)
                    hidden_states = residual + moe_output
                else:
                    hidden_states = residual + layer.mlp(hidden_states_for_router)
                
                # 确保hidden_states保持bfloat16类型
                if hidden_states.dtype != torch.bfloat16:
                    hidden_states = hidden_states.to(torch.bfloat16)
            
            # 释放当前层内存
            del layer, residual, hidden_states_for_router
            del unique_experts, selected_experts, routing_weights, router_logits
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
        
        # 打印专家使用统计
        print("\n专家使用统计:")
        total_experts_needed = sum(stats["selected_experts_count"] for stats in experts_stats.values())
        total_experts_available = self.config.num_hidden_layers * self.config.num_experts
        overall_memory_saving = 1.0 - (total_experts_needed / total_experts_available)
        
        print(f"总共加载了 {total_experts_needed}/{total_experts_available} 个专家")
        print(f"总体内存节省率: {overall_memory_saving*100:.2f}%")
        print(f"理论上节省了 {overall_memory_saving*100:.2f}% 的专家权重内存占用")
        
        # 返回结果
        return {
            "input_text": output_text,
            "predicted_token": predicted_token,
            "hidden_states": hidden_states,
            "logits": logits,
            "total_time": total_time,
            "experts_stats": experts_stats,
            "overall_memory_saving": overall_memory_saving
        }
    
    def generate_text(self, prompt, max_new_tokens=20):
        """使用选择性专家加载方式生成文本"""
        print(f"\n开始生成文本，使用选择性专家加载，输入: \"{prompt}\"")
        print_memory_usage("生成开始")
        
        # 记录开始时间
        start_time = time.time()
        
        # 对输入进行编码
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        generated_ids = input_ids.clone()
        
        # 收集专家使用统计
        experts_stats_per_token = []
        
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
            
            # 收集当前token的专家统计
            token_experts_stats = {}
            
            # 依次处理每一层
            for layer_idx in range(self.config.num_hidden_layers):
                # 第一阶段：创建空层，加载注意力层、归一化层和路由器权重
                layer = self._create_empty_layer(layer_idx)
                self._load_attention_and_norm(layer, layer_idx)
                self._load_router_weights(layer, layer_idx)
                
                # 运行注意力机制的前向传播
                with torch.no_grad():
                    # 执行自注意力部分的计算
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)
                    
                    # 应用自注意力
                    hidden_states = layer.self_attn(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                    )[0]
                    
                    hidden_states = residual + hidden_states
                
                # 第二阶段：确定需要加载哪些专家
                # 运行路由器获取专家选择
                residual = hidden_states
                hidden_states_for_router = layer.post_attention_layernorm(hidden_states)
                
                unique_experts, selected_experts, routing_weights, router_logits = self._run_router_get_experts(layer, hidden_states_for_router)
                
                # 记录专家使用情况
                token_experts_stats[f"layer_{layer_idx}"] = {
                    "unique_experts": unique_experts.tolist(),
                    "selected_experts_count": len(unique_experts),
                    "total_experts": self.config.num_experts,
                    "memory_saving_ratio": 1.0 - (len(unique_experts) / self.config.num_experts)
                }
                
                # 第三阶段：只加载需要的专家权重
                for expert_idx in unique_experts:
                    self._load_expert_weights(layer, layer_idx, expert_idx)
                
                # 第四阶段：执行MLP (MOE) 的前向传播
                with torch.no_grad():
                    # 执行MLP (MOE) 部分
                    # 只传递hidden_states_for_router，不传递router_logits
                    if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                        moe_output, _ = layer.mlp(hidden_states_for_router)
                        hidden_states = residual + moe_output
                    else:
                        hidden_states = residual + layer.mlp(hidden_states_for_router)
                    
                    # 确保hidden_states保持bfloat16类型
                    if hidden_states.dtype != torch.bfloat16:
                        hidden_states = hidden_states.to(torch.bfloat16)
                
                # 释放当前层内存
                del layer, residual, hidden_states_for_router
                del unique_experts, selected_experts, routing_weights, router_logits
                clean_memory()
            
            # 记录当前token的专家统计
            experts_stats_per_token.append(token_experts_stats)
            
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
            
            # 计算当前token的专家使用统计
            token_total_experts_needed = sum(stats["selected_experts_count"] for stats in token_experts_stats.values())
            token_total_experts_available = self.config.num_hidden_layers * self.config.num_experts
            token_memory_saving = 1.0 - (token_total_experts_needed / token_total_experts_available)
            
            print(f"生成的token: {current_token} (用时: {token_time:.2f}秒, 专家内存节省: {token_memory_saving*100:.2f}%)")
            
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
        
        # 计算总用时和平均专家使用统计
        total_time = time.time() - start_time
        
        # 计算整体专家使用统计
        token_count = len(experts_stats_per_token)
        total_experts_needed = sum(
            sum(stats["selected_experts_count"] for stats in token_stats.values())
            for token_stats in experts_stats_per_token
        )
        total_experts_available = token_count * self.config.num_hidden_layers * self.config.num_experts
        overall_memory_saving = 1.0 - (total_experts_needed / total_experts_available)
        
        print(f"\n生成完成，总用时: {total_time:.2f}秒，平均每token: {total_time/token_count:.2f}秒")
        print(f"总体专家内存节省率: {overall_memory_saving*100:.2f}%")
        print(f"生成的完整文本: {output_text}")
        print_memory_usage("生成结束")
        
        return {
            "generated_text": output_text,
            "total_time": total_time,
            "tokens_generated": token_count,
            "experts_stats_per_token": experts_stats_per_token,
            "overall_memory_saving": overall_memory_saving
        }

def main():
    # 设置模型路径
    model_path = "/home/szm/qwen3_a3b"  # 修改为MOE模型路径
    experts_dir = "moe_experts"  # 修改为MOE专家权重分割目录
    
    # 打印初始内存使用情况
    print("初始内存使用情况:")
    print_memory_usage("程序开始")
    
    try:
        # 初始化推理器 - 强制使用CPU
        inferencer = SelectiveExpertsMoeInference(model_path, experts_dir=experts_dir, device="cpu")
        
        # 设置测试输入
        prompt = "介绍一下大语言模型"
        
        # 方法1: 处理所有层并获取下一个token预测
        print("\n===== 方法1: 选择性专家加载方式处理所有层 =====")
        result = inferencer.process_all_layers(prompt)
        
        # 方法2: 生成多个token形成文本
        print("\n===== 方法2: 选择性专家加载方式生成文本 =====")
        generation_result = inferencer.generate_text(prompt, max_new_tokens=20) # 减少token数量以加快测试
        
        print("\n测试完成!")
        print(f"生成的文本: {generation_result['generated_text']}")
        print(f"总体专家内存节省率: {generation_result['overall_memory_saving']*100:.2f}%")
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