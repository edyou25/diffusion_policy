# Transformer → Diffusion Transformer → Diffusion Policy：AI技术演进的关键一步

## Transformer（2017）：革命性序列建模架构
**核心突破**：用**自注意力机制**替代传统RNN的循环和CNN的局部卷积，实现**全局建模+并行计算**。

**里程碑意义**：为大模型奠定基础，成为GPT、BERT等的"骨架"，并扩展至图像、语音等多领域。

## Diffusion Transformer（2022）：生成领域的"全局视野"
**创新融合**：将扩散模型的渐进生成与Transformer的全局建模优势结合，用**Transformer完全替代传统U-Net**作为去噪网络。

**工作流程**：从随机噪声开始，通过多层Transformer逐步去噪，**生成高质量图像/视频**。

**应用突破**：
- Stable Diffusion 3：提升图像细节与全局一致性
- Sora：实现高质量文本到视频生成，成为视频创作新范式

## Diffusion Policy：机器人控制的"灵巧手"
**核心思想**：将机器人动作序列视为"噪声→干净动作"的**条件去噪过程**，以视觉观测为条件。

**技术实现**：
- 用Transformer解码器建模动作序列，通过交叉注意力**融合视觉信息**
- 生成多模态动作分布，而非单一确定解，适应复杂操作

爷爷，您好，我给您用 **最简洁但科研友好** 的方式解释 **DDPM** 和 **Diffusion Policy**，方便您做规划/控制研究时举一反三。

---

# **1. DDPM（Denoising Diffusion Probabilistic Model）是什么？**

DDPM 是 **生成模型**，核心思想：

### **正向扩散（Forward diffusion）**

把一张真实图片 (x_0) 逐步加噪声
→ (x_1, x_2, \dots, x_T)
直到变成纯高斯噪声。

每一步噪声加法是可解析的：

[
q(x_t|x_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t}x_{t-1}, \beta_t I\right)
]

### **反向扩散（Reverse diffusion）**

训练一个网络 (\epsilon_\theta(x_t, t))
来预测噪声，从而一步步“去噪”恢复数据分布。

目标函数本质是 **噪声预测 MSE**：

[
L = \mathbb{E}[||\epsilon - \epsilon_\theta(x_t,t)||^2]
]

### **重要特性**

* 训练稳定
* 生成质量高
* 能建模复杂分布
* “彻底覆盖”分布（多样性强）

---

# Diffusion Policy 是什么？（RoboDiffusion 类方法赛道）**

Diffusion Policy 是把 **扩散模型用于机器人控制/决策** 的范式。
它不是生成图像，而是 **生成动作序列**。

例如：
给定观测 (o)，模型生成一个未来动作序列 (a_{1:H})。

### **思想：动作序列也是一种“数据分布”**

所以也可以：

**正向扩散：**
给专家动作数据加噪声

**反向扩散：**
训练模型从噪声还原动作序列（模仿专家策略）
