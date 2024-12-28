# @Time    : 28/12/2024 上午 10:00
# @Author  : Xuan
# @File    : 基于深度可分离膨胀卷积的MNIST手写体识别.py
# @Software: PyCharm


"""
项目背景：
MNIST手写体数据集是深度学习中一个经典的入门数据集，用于评估不同模型在图像分类任务上的性能。
在本项目中，我们设计并实现了一种基于深度可分离膨胀卷积的神经网络模型，以减少模型参数量，同时提升模型的特征提取能力和识别效果。

核心技术：
1. 深度卷积：将标准卷积操作分解为通道维度的独立卷积，大幅减少计算开销。
2. 点卷积：通过1x1卷积整合通道间的信息，弥补深度卷积的表达能力不足。
3. 膨胀卷积：扩展卷积核的感受野，在不增加参数的情况下捕获更大范围的上下文信息。

项目实现：
下面的代码展示了整个流程，从数据加载与预处理到模型定义、训练和测试，最后对深度可分离卷积与标准卷积的参数量进行对比分析。
"""

import torch
import torch.nn as nn
import einops.layers.torch as elt
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载训练集
train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 下载并加载测试集
test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 检查数据集大小
print(f'Train dataset size: {len(train_dataset)}')  # Train dataset size: 60000
print(f'Test dataset size: {len(test_dataset)}')  # Test dataset size: 10000


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 深度、可分离、膨胀卷积
        self.conv = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=7),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, groups=6, dilation=2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(24, 6, kernel_size=7),
        )
        self.logits_layer = nn.Linear(in_features=6 * 12 * 12, out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = elt.Rearrange('b c h w -> b (c h w)')(x)
        logits = self.logits_layer(x)
        return logits


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), './modelsave/mnist.pth')

# Load the model
model = Model().to(device)
model.load_state_dict(torch.load('./modelsave/mnist.pth'))

# Evaluation
model.eval()
correct = 0
first_image_displayed = False
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Display the first image and its prediction
        if not first_image_displayed:
            plt.imshow(data[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title(f'Predicted: {pred[0].item()}')
            plt.show()
            first_image_displayed = True

print(f'Test set: Accuracy: {correct / len(test_loader.dataset):.4f}')  # Test set: Accuracy: 0.9874

# 深度可分离卷积参数比较
# 普通卷积参数量
conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)  # in(3) * out(3) * k(3) * k(3) + out(3) = 84
conv_params = sum(p.numel() for p in conv.parameters())
print('conv_params:', conv_params)  # conv_params: 84

# 深度可分离卷积参数量
depthwise = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3)  # in(3) * k(3) * k(3) + out(3) = 30
pointwise = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)  # in(3) * out(3) * k(1) * k(1) + out(3) = 12
depthwise_separable = nn.Sequential(depthwise, pointwise)
depthwise_separable_params = sum(p.numel() for p in depthwise_separable.parameters())
print('depthwise_separable_params:', depthwise_separable_params)  # depthwise_separable_params: 42
