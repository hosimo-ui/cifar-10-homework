import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()
        
        # 使用更大的通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate/2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 自适应池化
            nn.Dropout2d(dropout_rate)
        )
        
        # 添加残差连接或SE模块
        self.se1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(96, 96//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(96//16, 96, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(384 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        
        # SE注意力
        se = self.se1(x)
        x = x * se
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x