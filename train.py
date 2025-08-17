import os
import argparse
import torch
import tqdm
from sklearn import metrics
import torch.nn as nn
import utils.utils as u
from iden_modules.modeling.model import CLIPRModel, VisionModel, TextModel, ProjectionLayer
from utils.dataset_finetuning import CusImageDataset
from torch.utils.data import DataLoader
import glob
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def calculate_metrics(labels, outputs):
    """计算准确率、精确率、召回率、F1值"""
    Acc = metrics.accuracy_score(labels, outputs)
    precision = precision_score(labels, outputs, average="macro")
    recall = recall_score(labels, outputs, average="macro")
    f1 = f1_score(labels, outputs, average="macro")
    return Acc, precision, recall, f1


def plot_confusion_matrix(labels, outputs, title, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, outputs)
    true_labels = np.unique(labels)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(true_labels)), true_labels)
    plt.yticks(np.arange(len(true_labels)), true_labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(title)

    for i in range(len(true_labels)):
        for j in range(len(true_labels)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

    plt.savefig(save_path)
    plt.close()


def save_detailed_class_report(epoch, class_info, save_dir, mode):
    """保存详细的类别准确率报告"""
    report_path = os.path.join(save_dir, f"{mode}_class_report_epoch_{epoch}.csv")
    with open(report_path, 'w') as f:
        f.write("类别,准确率,样本数\n")
        for class_label in sorted(class_info.keys()):
            f.write(f"{class_label},{class_info[class_label]['accuracy']:.4f},{class_info[class_label]['count']}\n")
    print(f"已保存 {mode} 集类别详细报告到 {report_path}")


def val(val_dataloader, model, epoch, args, mode):
    """验证/测试函数，返回总体准确率和类别详细信息"""
    print(f'\n====== 开始 {mode} ======!')
    model.eval()
    labels = []
    outputs = []

    tbar = tqdm.tqdm(val_dataloader, desc='\r')
    with torch.no_grad():
        for img_data_list in tbar:
            Fundus_img = img_data_list[0].cuda()
            cls_label = img_data_list[1].long().cuda()
            pred = model.forward(Fundus_img)
            pred = torch.softmax(pred, dim=1)
            pred_decision = pred.argmax(dim=-1)

            outputs.extend(pred_decision.tolist())
            labels.extend(cls_label.tolist())

    # 计算总体指标
    Acc, precision, recall, f1 = calculate_metrics(labels, outputs)

    # 保存总体指标
    log_dir = f"logs/{mode}"
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/result-{epoch}.txt", "a") as f:
        f.write(f"\nEpoch {epoch} - {mode}:\n")
        f.write(f"准确率: {Acc * 100:.2f}%\n")
        f.write(f"精确率: {precision * 100:.2f}%\n")
        f.write(f"召回率: {recall * 100:.2f}%\n")
        f.write(f"F1值: {f1 * 100:.2f}%\n")

    print(f"Epoch {epoch} - {mode} - 准确率: {Acc * 100:.2f}%")
    print(f"Epoch {epoch} - {mode} - 精确率: {precision * 100:.2f}%")
    print(f"Epoch {epoch} - {mode} - 召回率: {recall * 100:.2f}%")
    print(f"Epoch {epoch} - {mode} - F1值: {f1 * 100:.2f}%")

    # 绘制混淆矩阵
    plot_confusion_matrix(labels, outputs, f'混淆矩阵 - {mode} - Epoch {epoch}', f"{log_dir}/cm-{epoch}.jpg")

    # 计算每个类别的准确率
    unique_labels = np.unique(labels)
    class_info = {}

    for class_label in unique_labels:
        class_mask = np.array(labels) == class_label
        class_correct = np.array(outputs)[class_mask] == class_label
        class_accuracy = np.mean(class_correct) if sum(class_mask) > 0 else 0.0
        class_info[class_label] = {
            'accuracy': class_accuracy,
            'count': sum(class_mask)
        }
        print(f"类别 {class_label}: 准确率={class_accuracy:.4f}, 样本数={sum(class_mask)}")

    # 保存到总体记录文件
    os.makedirs(os.path.join(args.save_model_path, args.net_work), exist_ok=True)
    summary_file = os.path.join(args.save_model_path, args.net_work, "accuracy_summary.csv")

    if not os.path.exists(summary_file):
        with open(summary_file, 'w') as f:
            f.write("epoch,mode,class,accuracy,count\n")

    with open(summary_file, 'a') as f:
        for class_label in sorted(class_info.keys()):
            f.write(f"{epoch},{mode},{class_label},{class_info[class_label]['accuracy']:.4f},{class_info[class_label]['count']}\n")

    torch.cuda.empty_cache()
    return Acc, class_info


def calculate_class_weights(train_csv, num_classes):
    """计算类别权重（样本数越少，权重越高）"""
    df = pd.read_csv(train_csv)
    class_counts = df.iloc[:, 1].value_counts().sort_index().tolist()
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    return class_weights / class_weights.sum()  # 归一化权重



class CLIPRModelWithWeightedContrastive(CLIPRModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def softce_clip_loss(self, logits_per_text, target_pseudo, weight=None):
        """带权重的对比损失：文本→图像 + 图像→文本，均应用类别权重"""
        # 文本→图像的交叉熵
        caption_loss = F.cross_entropy(logits_per_text, target_pseudo, weight=weight)
        # 图像→文本的交叉熵
        image_loss = F.cross_entropy(logits_per_text.T, target_pseudo, weight=weight)
        return (caption_loss + image_loss) / 2.0  # 双向平均


def train(train_loader, val_loader, test_loader, model, optimizer, args, weight_path, csv_path, class_weights):
    """训练函数 - 对比损失加入类别权重，同时保持与CLIPRModel核心逻辑一致"""
    model = model.cuda()
    best_Acc = 0.0
    class_report_dir = os.path.join(args.save_model_path, args.net_work, "class_reports")
    os.makedirs(class_report_dir, exist_ok=True)

    # 实例化带权重对比损失的CLIPRModel
    clip_model = CLIPRModelWithWeightedContrastive(
        vision_type="lora", 
        from_checkpoint=True,
        weights_path=weight_path, 
        R=8,
        bert_type="./BioBert"
    ).cuda()
    clip_model.eval()  # 仅用其损失方法，不更新参数

    # 预生成类别文本特征（与CLIPRModel文本处理逻辑一致）
    class_names = ["Proliferative Diabetic Retinopathy", "Retinitis Pigmentosa", "Vitreoretinal Lymphoma",
                   " Vogt-Koyanagi-Harada Disease", "dry Age-related Macular Degeneration",
                   "myopic Choroidal Neovascularization"]
    text_prompts = [f"An OCT image of {name}" for name in class_names]
    text_tokens = clip_model.text_model.tokenize(text_prompts)
    with torch.no_grad():
        text_features = clip_model.text_model(
            text_tokens["input_ids"].cuda(),
            text_tokens["attention_mask"].cuda()
        )  # 形状 [6, proj_dim]

    # 从第58轮开始训练
    start_epoch = 58

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        train_loss = 0.0
        labels = []
        outputs = []

        tq = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}')
        for img_data_list in tq:
            Fundus_img = img_data_list[0].cuda()  # [batch_size, 3, H, W]
            cls_label = img_data_list[1].long().cuda()  # [batch_size]

            optimizer.zero_grad()

            # 1. 获取图像特征和分类logits
            img_features = model.img_encoder(Fundus_img)  # (batch_size, 1024)
            pred_logits = model.classifier(img_features)  # (batch_size, num_classes)

            # 2.对比损失（
            # 2.1 图像特征投影
            img_features_proj = clip_model.vision_model.projection_head_vision(img_features)  # [batch_size, proj_dim]
            # 2.2 计算相似度矩阵
            logits_per_image = clip_model.compute_logits(img_features_proj, text_features)  # [batch_size, 6]
            logits_per_text = logits_per_image.t()  # [6, batch_size]（文本→图像的相似度）
            # 2.3 调用带权重的双向对比损失（传入类别权重）
            contrastive_loss = clip_model.softce_clip_loss(
                logits_per_text, 
                cls_label, 
                weight=class_weights  # 核心：对比损失加入类别权重
            )

            # 3. 不确定性损失
            evidences = F.softplus(pred_logits)  
            alpha = evidences + 1
            uncertainty_loss = clip_model.un_ce_loss(
                p=cls_label, 
                alpha=alpha, 
                c=args.num_classes, 
                global_step=epoch, 
                annealing_step=15
            ).mean()

            # 4. 总损失：对比损失（带权重）+ 不确定性损失
            total_loss = contrastive_loss + uncertainty_loss

            # 反向传播与优化
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

            # 记录损失与预测结果
            train_loss += total_loss.item()
            tq.set_postfix(loss=f'{train_loss / (tq.n + 1):.4f}')
            pred_softmax = torch.softmax(pred_logits, dim=1)
            outputs.extend(torch.argmax(pred_softmax, dim=-1).tolist())
            labels.extend(cls_label.tolist())

        # 计算训练指标
        train_Acc, train_precision, train_recall, train_f1 = calculate_metrics(labels, outputs)
        print(f"\nEpoch {epoch} 训练结果:")
        print(f"准确率: {train_Acc * 100:.2f}% | 精确率: {train_precision * 100:.2f}%")
        print(f"召回率: {train_recall * 100:.2f}% | F1值: {train_f1 * 100:.2f}%")
        print(f"带权重对比损失: {contrastive_loss.item():.3f} | 不确定性损失: {uncertainty_loss.item():.3f}")

        # 验证与测试
        val_Acc, val_class_info = val(val_loader, model, epoch, args, "val")
        test_Acc, test_class_info = val(test_loader, model, epoch, args, "test")

        # 保存报告与模型
        save_detailed_class_report(epoch, val_class_info, class_report_dir, "val")
        save_detailed_class_report(epoch, test_class_info, class_report_dir, "test")
        checkpoint_dir = os.path.join(args.save_model_path, args.net_work)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 保存当前轮模型
        u.save_checkpoint_epoch({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'mean_ACC': val_Acc,
        }, epoch, True, checkpoint_dir, "Train",
                               filename=os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth.tar"))
        # 保存最佳模型
        if val_Acc > best_Acc:
            best_Acc = val_Acc
            best_checkpoint_path = os.path.join(args.save_model_path, "best_checkpoint.pth.tar")
            if os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)
            u.save_checkpoint_epoch({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'mean_ACC': val_Acc,
            }, epoch, True, args.save_model_path, "Test",
                                   filename=best_checkpoint_path)
            print(f"发现新的最佳模型，验证准确率: {val_Acc * 100:.2f}%")


class Model_Finetuing(torch.nn.Module):
    def __init__(self, model_name, class_num, weight_path):
        super().__init__()
        Model_Pretrained = CLIPRModel(
            vision_type=model_name, 
            from_checkpoint=True,
            weights_path=weight_path, 
            R=8,
            bert_type="./BioBert"
        )
        self.img_encoder = Model_Pretrained.vision_model.model
        feature_dim = 512
        self.classifier = torch.nn.Linear(feature_dim, class_num, bias=True)

    def forward(self, x):
        x_features = self.img_encoder(x)
        return self.classifier(x_features)


def main(args):
    # 1. 初始化模型
    weight_path = "./BioBert/RetiZero.pth"
    model = Model_Finetuing(model_name="lora", class_num=args.num_classes, weight_path=weight_path)

    # 2. 加载预训练权重
    pretrained_weights_path = "/public/home/pyy_www_2706/fenlei/RetiZero-main2/RetiZero-main3/Model_saved_UA5e-4quancheng/lora/model_Train_067.pth.tar"
    checkpoint = torch.load(pretrained_weights_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])

    # 3. 计算类别权重（用于带权重的对比损失）
    data_path = args.data_path
    csv_path = args.csv_path
    train_csv = os.path.join(csv_path, "train_0.csv")
    class_weights = calculate_class_weights(train_csv, args.num_classes).cuda()
    print("类别权重（样本少的类别权重更高）:", class_weights)

    # 4. 数据加载
    train_dataset = CusImageDataset(csv_file=train_csv, data_path=os.path.join(data_path, "train"))
    val_dataset = CusImageDataset(csv_file=os.path.join(csv_path, "val_0.csv"), data_path=os.path.join(data_path, "val"))
    test_dataset = CusImageDataset(csv_file=os.path.join(csv_path, "test_0.csv"), data_path=os.path.join(data_path, "test"))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 5. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6. 开始训练（传入类别权重）
    train(train_loader, val_loader, test_loader, model, optimizer, args, weight_path, csv_path, class_weights)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_model_path", default="./Model_saved_UA5e-4quancheng")
    parser.add_argument("--net_work", default="lora")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--csv_path", default="/public/home/pyy_www_2706/fenlei/oct_data/data/csv_all/6")
    parser.add_argument("--data_path", default="/public/home/pyy_www_2706/fenlei/oct_data/data")
    return parser


if __name__ == "__main__":
    torch.set_num_threads(4)
    args = get_parser().parse_args()
    args.seed = 1234
    u.setup_seed(args.seed)

    print("训练参数配置:")
    print(f"类别数: {args.num_classes}")
    print(f"训练周期: {args.num_epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"模型保存路径: {args.save_model_path}")
    print(f"数据CSV路径: {args.csv_path}")
    print(f"图像数据路径: {args.data_path}")

    main(args)
