<img width="814" height="458" alt="image" src="https://github.com/user-attachments/assets/2fb2fbb9-43aa-4214-8d35-505d21b75ee1" />
<img width="813" height="333" alt="image" src="https://github.com/user-attachments/assets/4b0f89ec-5980-499f-9308-9d625bc45fae" />


# UA-VLFM




模型采用基于RETFound结合LoRA优化的图像编码器及基于BioClinicalBERT的文本编码器；结合CLIP对比学习，确定正负样本对，拉近正样本对的距离；
模型采用不确定性估计方法，基于 Dirichlet分布计算置信质量和不确定性分数，结合改进的Youden's Index 确定阈值；
模型训练时，采用类平衡损失与不确定性损失的总损失函数指导模型优化
推理阶段输出诊断结果及不确定性分数
