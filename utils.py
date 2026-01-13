import torch
import os
from config import config

def save_checkpoint(state, filename='checkpoint.pth'):
    """保存模型checkpoint"""
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    
    filepath = os.path.join(config.save_dir, filename)
    torch.save(state, filepath)
    print(f"模型已保存到: {filepath}")

def load_checkpoint(model, optimizer=None, scheduler=None, filename='checkpoint.pth'):
    """加载模型checkpoint"""
    filepath = os.path.join(config.save_dir, filename)
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location=config.device)
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"已加载checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    else:
        print(f"未找到checkpoint: {filepath}")
        return 0