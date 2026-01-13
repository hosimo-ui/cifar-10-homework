import torch
import numpy as np
from models.simple_cnn import SimpleCNN
from dataset import get_dataloaders, get_classes
from config import config

def test(model_path=None):
    """测试函数"""
    
    # 获取数据
    _, test_loader = get_dataloaders()
    classes = get_classes()
    
    # 加载模型
    model = SimpleCNN(config.num_classes)
    model = model.to(config.device)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=config.device)
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    print("开始测试...")
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 每个类别的准确率
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # 打印总体准确率
    print(f'\n测试集准确率: {100 * correct / total:.2f}%')
    
    # 打印每个类别的准确率
    print('\n各个类别的准确率:')
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'{classes[i]:12s}: {accuracy:.2f}%')
    
    return 100 * correct / total

if __name__ == "__main__":
    # 可以在这里指定要加载的模型路径
    # accuracy = test("checkpoints/cifar10_cnn.pth")
    accuracy = test("checkpoints/best_model.pth")