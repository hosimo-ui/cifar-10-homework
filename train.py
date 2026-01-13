import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from models.simple_cnn import SimpleCNN
from dataset import get_dataloaders
from config import config
from utils import save_checkpoint
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

def validate(model, test_loader, criterion):
    """验证函数"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.device, non_blocking=True)
            targets = targets.to(config.device, non_blocking=True)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """绘制训练曲线"""
    if not os.path.exists(config.plot_dir):
        os.makedirs(config.plot_dir)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(config.plot_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"训练曲线已保存到: {plot_path}")

def train():
    """训练函数"""
    
    # 获取数据
    train_loader, test_loader = get_dataloaders()
    
    # 初始化模型
    model = SimpleCNN(config.num_classes)
    model = model.to(config.device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                         lr=config.learning_rate,
                         weight_decay=config.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                   T_max=config.epochs)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 记录训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 最佳模型追踪
    best_acc = 0.0
    
    # 训练循环
    print(f"开始训练，使用设备: {config.device}")
    print(f"批次大小: {config.batch_size}, 工作进程: {config.num_workers}")
    print(f"混合精度训练: 已启用")
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(config.device, non_blocking=True)
            targets = targets.to(config.device, non_blocking=True)
            
            # 前向传播（混合精度）
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 反向传播（混合精度）
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': running_loss/(batch_idx+1),
                'Acc': 100.*correct/total,
                'LR': optimizer.param_groups[0]['lr']
            })
        
        # 记录训练指标
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc = validate(model, test_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{config.epochs}:')
        print(f'  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(config.save_dir, config.best_model_name)
            if not os.path.exists(config.save_dir):
                os.makedirs(config.save_dir)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
            }, best_model_path)
            print(f'  ✓ 新的最佳模型已保存 (准确率: {best_acc:.2f}%)')
        
        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
            }, filename=f'checkpoint_epoch_{epoch+1}.pth')
        
        # 绘制训练曲线
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    print("\n训练完成！")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    
    # 最终绘制训练曲线
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    return model

if __name__ == "__main__":
    train()
