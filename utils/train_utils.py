# 训练辅助函数 (Warmup, Cosine Decay)
        
        def get_optimizer(model, lr):
            # 仅优化需要梯度的参数
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            # 实际使用 AdamW 或 SGD
            return torch.optim.Adam(trainable_params, lr=lr)
        
        def get_scheduler(optimizer, total_steps):
            # 学习率调度器 (例如 Cosine Decay)
            # ... 实现 Warmup 和 Cosine Decay 逻辑 ...
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        