import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# 导入模型定义和数据加载类（需与训练代码保持一致）
from iden_modules.modeling.model import CLIPRModel
from utils.dataset_finetuning import CusImageDataset
import utils.utils as u

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class Model_Finetuing(torch.nn.Module):
    """与训练代码中一致的模型定义"""
    def __init__(self, model_name, class_num, weight_path, lora_rank=8):
        super().__init__()
        Model_Pretrained = CLIPRModel(
            vision_type=model_name, 
            from_checkpoint=True,
            weights_path=weight_path, 
            R=lora_rank
        )
        self.img_encoder = Model_Pretrained.vision_model.model
        # 根据模型类型确定特征维度
        self.feature_dim = 512 if model_name == "lora" else 1024
        self.classifier = torch.nn.Linear(self.feature_dim, class_num, bias=True)

    def forward(self, x):
        x_features = self.img_encoder(x)
        return self.classifier(x_features)


def calculate_metrics(labels, outputs):
    """计算评估指标"""
    Acc = accuracy_score(labels, outputs)
    precision = precision_score(labels, outputs, average="macro")
    recall = recall_score(labels, outputs, average="macro")
    f1 = f1_score(labels, outputs, average="macro")
    return {
        "准确率": Acc * 100,
        "精确率": precision * 100,
        "召回率": recall * 100,
        "F1值": f1 * 100
    }


def plot_confusion_matrix(labels, outputs, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, outputs)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # 在混淆矩阵中标记数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('测试集混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_class_report(labels, outputs, class_names, save_path):
    """保存每个类别的详细评估报告"""
    class_metrics = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = np.array(labels) == label
        total = sum(mask)
        correct = sum(np.array(outputs)[mask] == label)
        acc = (correct / total) * 100 if total > 0 else 0
        
        class_metrics[class_names[label]] = {
            "类别标签": label,
            "样本数": total,
            "正确预测数": correct,
            "准确率(%)": round(acc, 2)
        }
    
    # 保存为CSV
    df = pd.DataFrame.from_dict(class_metrics, orient='index')
    df.to_csv(save_path, encoding='utf-8-sig')
    return class_metrics


def test(args):
    # 类别名称（与训练时保持一致）
    class_names = ["PDR", "RP", "VRL", "acute VKH", "dAMD", "mCNV"]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model = Model_Finetuing(
        model_name="lora",
        class_num=args.num_classes,
        weight_path=args.pretrained_weight_path,
        lora_rank=args.lora_rank
    ).to(device)
    
    # 加载最佳模型权重
    checkpoint = torch.load(args.best_model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"成功加载最佳模型（epoch {checkpoint['epoch']}，验证准确率: {checkpoint['mean_ACC']*100:.2f}%）")
    
    # 加载测试数据
    print("加载测试数据...")
    test_dataset = CusImageDataset(
        csv_file=os.path.join(args.csv_path, "test_0.csv"),
        data_path=os.path.join(args.data_path, "test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 测试模型
    print("开始测试...")
    model.eval()
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for img_data_list in tqdm(test_loader, desc="测试进度"):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().cpu().numpy()
            
            # 模型预测
            pred = model.forward(Fundus_img)
            pred_probs = torch.softmax(pred, dim=1)
            pred_labels = torch.argmax(pred_probs, dim=-1).cpu().numpy()
            
            all_labels.extend(cls_label)
            all_outputs.extend(pred_labels)
    
    # 计算评估指标
    metrics = calculate_metrics(all_labels, all_outputs)
    print("\n====== 测试集总体评估指标 ======")
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}%")
    
    # 保存总体指标
    with open(os.path.join(args.output_dir, "test_summary.txt"), "w") as f:
        f.write("测试集评估结果\n")
        f.write(f"测试样本数: {len(all_labels)}\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.2f}%\n")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_outputs, class_names, cm_path)
    print(f"混淆矩阵已保存至: {cm_path}")
    
    # 保存类别详细报告
    class_report_path = os.path.join(args.output_dir, "class_report.csv")
    class_metrics = save_class_report(all_labels, all_outputs, class_names, class_report_path)
    print(f"类别详细报告已保存至: {class_report_path}")
    
    # 打印每个类别的准确率
    print("\n====== 各类别准确率 ======")
    for class_name, metrics in class_metrics.items():
        print(f"{class_name}: {metrics['准确率(%)']}% (样本数: {metrics['样本数']})")


def get_parser():
    parser = argparse.ArgumentParser(description="模型测试脚本")
    parser.add_argument("--num_classes", type=int, default=6, help="类别数量")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--batch_size", type=int, default=64, help="测试批次大小")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA秩参数，需与训练时一致")
    
    # 文件路径配置
    parser.add_argument("--best_model_path", default="./Model_saved_UA5e-4_R8/best_checkpoint.pth.tar", 
                      help="最佳模型权重路径")
    parser.add_argument("--pretrained_weight_path", default="./BioBert/RETFound_oct_weights.pth",
                      help="预训练模型权重路径")
    parser.add_argument("--data_path", default="/public/home/pyy_www_2706/fenlei/oct_data/data",
                      help="数据根目录")
    parser.add_argument("--csv_path", default="/public/home/pyy_www_2706/fenlei/oct_data/data/csv_all/6",
                      help="CSV文件目录")
    parser.add_argument("--output_dir", default="./test_results/R8", 
                      help="测试结果输出目录")
    
    return parser


if __name__ == "__main__":
    # 设置随机种子
    u.setup_seed(1234)
    # 解析参数
    args = get_parser().parse_args()
    
    # 打印配置信息
    print("测试配置:")
    print(f"最佳模型路径: {args.best_model_path}")
    print(f"LoRA秩: {args.lora_rank}")
    print(f"测试数据路径: {args.data_path}")
    print(f"结果输出目录: {args.output_dir}")
    
    # 执行测试
    test(args)
    print("测试完成！")