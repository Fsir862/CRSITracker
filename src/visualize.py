
import os
import sys
import torch
import numpy as np
from torchsummaryX import summary
from torchviz import make_dot
import matplotlib.pyplot as plt
import argparse

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from lib.model.networks.dla import DLASeg

# 自定义参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Model Visualization Script')
    parser.add_argument('--input_size', type=int, default=256, 
                        help='Input image size for visualization')
    parser.add_argument('--skip_pretrained', action='store_true',
                        help='Skip loading pretrained weights')
    return parser.parse_args([])  # 空列表避免真实命令行解析

# 创建默认配置对象
def setup_default_opt():
    class Opt:
        def __init__(self):
            # 必需参数
            self.task = 'tracking'
            self.load_model = 'skip_pretrained'
            self.mode = 'train'
            
            # 模型结构参数
            self.arch = 'dla_34'
            self.dla_node = 'dcn'
            self.head_conv = -1
            self.num_head_conv = 1
            self.head_kernel = 3
            self.down_ratio = 4
            self.not_idaup = False
            self.num_classes = -1
            self.num_layers = 34
            self.backbone = 'dla34'
            self.neck = 'dlaup'
            self.msra_outchannel = 256
            self.efficient_level = 0
            self.prior_bias = -4.6
            
            # 时序信息配置
            self.pre_hm = True
            self.pre_img = True
            self.pre_hm_method = "concat"
            self.no_pre_img = False
            self.no_prehm_input = False
            
            # 注意力机制配置
            self.atten_method = "reweight"
            self.atten_space = True
            self.atten_channel = True
            
            # 训练/推理配置
            self.inference_train = False
            self.shortnet = False
            self.lowfeat = False
            
            # 输出头配置
            self.heads = {'hm': 1, 'reg': 2, 'wh': 2}
            self.head_convs = {'hm': [256], 'reg': [256], 'wh': [256]}
            
            # 防止其他可能出现的属性错误
            self.debug = 0
            self.resume = False
            self.gpus = '0'
            self.batch_size = 1
            self.lr = 1e-4
            self.val_intervals = 5
            self.save_point = [90, 120]
            self.lr_step = [90, 120]
    
    return Opt()

def print_model_structure(model):
    """手动打印模型结构"""
    print("="*80)
    print("模型层级结构")
    print("="*80)
    
    def print_layer_info(name, module, indent=0):
        prefix = "  " * indent
        print(f"{prefix}{name} ({type(module).__name__})")
        
        # 打印参数信息
        if hasattr(module, 'weight') and module.weight is not None:
            print(f"{prefix}  Weight: {tuple(module.weight.shape)}")
        if hasattr(module, 'bias') and module.bias is not None:
            print(f"{prefix}  Bias: {tuple(module.bias.shape)}")
        
        # 递归打印子模块
        for child_name, child_module in module.named_children():
            print_layer_info(child_name, child_module, indent+1)
    
    for name, module in model.named_children():
        print_layer_info(name, module)
    
    print("="*80)

def visualize_model():
    # 获取自定义参数
    args = parse_args()
    
    # 创建配置对象
    opt = setup_default_opt()
    
    # 根据参数调整配置
    if args.skip_pretrained:
        opt.load_model = 'skip_pretrained'
    
    # 创建模型实例
    model = DLASeg(
        num_layers=34,
        heads=opt.heads,
        head_convs=opt.head_convs,
        opt=opt
    )
    model.eval()  # 设置为评估模式
    
    # 打印基础信息
    print(f"模型名称: {model.__class__.__name__}")
    print(f"输入通道: {model.base.channels[0]}")
    print(f"输出特征图数量: {len(model.base.channels)}")
    
    # 1. 手动打印模型结构
    print_model_structure(model)
    
    # 2. 使用 torchsummaryX 获取模型摘要
    print("\n[模型摘要]")
    input_size = args.input_size
    input_shape = (1, 3, input_size, input_size)
    pre_img_shape = (1, 3, input_size, input_size)
    pre_hm_shape = (1, 1, input_size, input_size)
    
    try:
        summary(
            model,
            torch.randn(*input_shape),
            torch.randn(*pre_img_shape),
            torch.randn(*pre_hm_shape)
        )
    except Exception as e:
        print(f"模型摘要生成失败: {e}")
        print("将跳过详细摘要，继续其他可视化")
    
    # 3. 可视化特征提取流程
    print("\n[特征提取流程]")
    visualize_feature_flow(model, input_shape, pre_img_shape, pre_hm_shape)
    
    # 4. 生成计算图
    print("\n[生成计算图]")
    generate_computation_graph(model, input_shape, pre_img_shape, pre_hm_shape)
    
    print("\n可视化完成! 请查看生成的图像")

def visualize_feature_flow(model, input_shape, pre_img_shape, pre_hm_shape):
    """可视化特征提取流程"""
    # 创建输入
    current_frame = torch.randn(*input_shape)
    prev_frame = torch.randn(*pre_img_shape)
    prev_heatmap = torch.randn(*pre_hm_shape)
    
    # 前向传播获取中间特征
    with torch.no_grad():
        # 骨干网络
        base_features = model.base(current_frame, prev_frame, prev_heatmap)
        
        # DLAUp 特征融合
        dlaup_features = model.dla_up(base_features)
        
        # IDAUp 特征细化
        idaup_input = []
        for i in range(model.last_level - model.first_level):
            idaup_input.append(dlaup_features[i].clone())
        model.ida_up(idaup_input, 0, len(idaup_input))
        final_feature = idaup_input[-1]
    
    # 可视化骨干网络特征
    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(base_features):
        if feat is not None:  # 跳过 None 值
            # 取第一个通道的平均值
            avg_feat = feat[0].mean(dim=0).cpu().numpy()
            plt.subplot(2, 3, i+1)
            plt.imshow(avg_feat, cmap='viridis')
            plt.title(f'Base Level {i} ({feat.shape[1]}x{feat.shape[2]}x{feat.shape[3]})')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('base_features.png')
    plt.close()
    
    # 可视化最终特征图
    plt.figure(figsize=(8, 6))
    if final_feature is not None:
        final_avg = final_feature[0].mean(dim=0).cpu().numpy()
        plt.imshow(final_avg, cmap='viridis')
        plt.title(f'Final Feature Map ({final_feature.shape[1]}x{final_feature.shape[2]}x{final_feature.shape[3]})')
        plt.axis('off')
        plt.savefig('final_feature.png')
        plt.close()

def generate_computation_graph(model, input_shape, pre_img_shape, pre_hm_shape):
    """生成计算图可视化"""
    # 创建输入
    current_frame = torch.randn(*input_shape)
    prev_frame = torch.randn(*pre_img_shape)
    prev_heatmap = torch.randn(*pre_hm_shape)
    
    # 前向传播
    with torch.no_grad():
        features = model.imgpre2feats(current_frame, prev_frame, prev_heatmap)
    
    # 生成计算图
    if features[0] is not None:
        dot = make_dot(
            features[0], 
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True
        )
        dot.format = 'png'
        dot.render('dlaseg_computation_graph', view=False)
        print("计算图保存为: dlaseg_computation_graph.png")
    else:
        print("无法生成计算图：特征输出为 None")

if __name__ == "__main__":
    visualize_model()
