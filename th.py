import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import utils.utils as u

def calculate_optimal_threshold(uncertainty_scores, uncertainty_labels):
    """
    计算最佳不确定性阈值
    通过最大化改进的Youden's Index变体ℓ(θ) = 2×TPR(θ)−FPR(θ)确定阈值
    
    参数:
        uncertainty_scores: 模型预测的不确定性分数列表
        uncertainty_labels: 真实的不确定性标签(1表示错误预测,0表示正确预测)
    
    返回:
        optimal_threshold: 最佳阈值
        tpr: 真阳性率数组
        fpr: 假阳性率数组
        thresholds: 阈值数组
        scores: 改进的Youden's Index分数数组
    """
    # 计算ROC曲线（FPR, TPR, 阈值）
    fpr, tpr, thresholds = roc_curve(
        uncertainty_labels, 
        uncertainty_scores
    )
    
    # 计算改进的Youden's Index变体: ℓ(θ) = 2×TPR(θ)−FPR(θ)
    scores = 2 * tpr - fpr
    
    # 找到最大化分数的阈值
    max_score_idx = np.argmax(scores)
    optimal_threshold = thresholds[max_score_idx]
    
    return optimal_threshold, tpr, fpr, thresholds, scores

def plot_roc_curve(tpr, fpr, thresholds, optimal_threshold, scores, save_path):
    """绘制ROC ROC曲线并标记最佳阈值点"""
    plt.figure(figsize=(10, 6))
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, label='ROC Curve', color='blue')
    
    # 标记最佳阈值点
    max_score_idx = np.argmax(scores)
    plt.scatter(
        fpr[max_score_idx],
        tpr[max_score_idx],
        color='red', 
        s=100, 
        label=f'Optimal Threshold: {optimal_threshold:.4f}\nScore: {scores[max_score_idx]:.4f}'
    )
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve with Improved Youden\'s Index Optimal Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_curve(thresholds, scores, optimal_threshold, save_path):
    """绘制改进的Youden's Index分数随阈值变化的曲线"""
    plt.figure(figsize=(10, 6))
    
    # 绘制分数曲线
    plt.plot(thresholds, scores, label='Improved Youden\'s Index (2×TPR−FPR)', color='green')
    
    # 标记最佳阈值点
    max_score_idx = np.argmax(scores)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--')
    plt.scatter(
        optimal_threshold,
        scores[max_score_idx],
        color='red', 
        s=100, 
        label=f'Optimal Threshold: {optimal_threshold:.4f}\nMax Score: {scores[max_score_idx]:.4f}'
    )
    
    plt.xlabel('Threshold')
    plt.ylabel('Score (2×TPR−FPR)')
    plt.title('Improved Youden\'s Index Score vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    # 设置随机种子确保可重复性
    u.setup_seed(args.seed)
    
    # 加载预测结果文件
    if not os.path.exists(args.result_file):
        raise FileNotFoundError(f"结果文件不存在: {args.result_file}")
    
    result_df = pd.read_csv(args.result_file)
    
    # 验证文件包含必要的列
    required_columns = ['True Label', 'Predicted Label', 'Uncertainty Score']
    for col in required_columns:
        if col not in result_df.columns:
            raise ValueError(f"结果文件缺少必要的列: {col}")
    
    # 计算不确定性标签 (1表示预测错误, 0表示预测正确)
    result_df['uncertainty_label'] = (result_df['True Label'] != result_df['Predicted Label']).astype(int)
    
    # 提取不确定性分数和标签
    uncertainty_scores = result_df['Uncertainty Score'].values
    uncertainty_labels = result_df['uncertainty_label'].values
    
    # 计算最佳阈值
    optimal_threshold, tpr, fpr, thresholds, scores = calculate_optimal_threshold(
        uncertainty_scores, 
        uncertainty_labels
    )
    
    # 找到最佳阈值对应的TPR和FPR
    max_score_idx = np.argmax(scores)
    best_tpr = tpr[max_score_idx]
    best_fpr = fpr[max_score_idx]
    best_score = scores[max_score_idx]
    
    # 输出结果
    print(f"最佳不确定性阈值: {optimal_threshold:.6f}")
    print(f"在该阈值下:")
    print(f"改进的Youden's Index分数 (2×TPR−FPR): {best_score:.6f}")
    print(f"真阳性率 (TPR): {best_tpr:.6f}")
    print(f"假阳性率 (FPR): {best_fpr:.6f}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存阈值
    with open(os.path.join(args args.output_dir, 'optimal_threshold.txt'), 'w') as f:
        f.write(f"optimal_threshold: {optimal_threshold:.6f}\n")
        f.write(f"improved_youden_score: {best_score:.6f}\n")
        f.write(f"tpr_at_optimal: {best_tpr:.6f}\n")
        f.write(f"fpr_at_optimal: {best_fpr:.6f}\n")
    
    # 绘制ROC曲线
    plot_roc_curve(
        tpr, 
        fpr, 
        thresholds, 
        optimal_threshold,
        scores,
        os.path.join(args.output_dir, 'roc_curve.png')
    )
    
    # 绘制分数随阈值变化曲线
    plot_score_curve(
        thresholds,
        scores,
        optimal_threshold,
        os.path.join(args.output_dir, 'score_vs_threshold.png')
    )
    
    print(f"结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用改进的Youden\'s Index计算不确定性最佳阈值')
    parser.add_argument('--result_file', type=str, required=True, 
                      help='包含预测结果的CSV文件路径,需包含True Label, Predicted Label和Uncertainty Score列')
    parser.add_argument('--output_dir', type=str, default='improved_youden_results', 
                      help='阈值计算结果保存目录')
    parser.add_argument('--seed', type=int, default=1234, 
                      help='随机种子,确保结果可重复')
    
    args = parser.parse_args()
    main(args)
    