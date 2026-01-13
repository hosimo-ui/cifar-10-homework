import torch

class Config:
    # 数据配置
    data_dir = "./data"
    batch_size = 256
    num_workers = 10
    prefetch_factor = 4
    pin_memory = True
    
    # 训练配置
    epochs = 100
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    
    # 模型配置
    num_classes = 10
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存配置
    save_dir = "./checkpoints"
    model_name = "cifar10_cnn.pth"
    best_model_name = "best_model.pth"
    plot_dir = "./plots"
    
config = Config()