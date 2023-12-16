import torch, math

# Define a linear learning rate schedule for the first phase
def lr_lambda_phase1(step, total_updates_phase1=1000):
    return min(1.0, step / total_updates_phase1)

# Define a half-cosine decay learning rate schedule for the second phase
def lr_lambda_phase2(step, total_updates_phase2=299000):
    step_tensor = torch.tensor(step, dtype=torch.float32)
    return 0.5 * (1 + torch.cos((step_tensor / total_updates_phase2) * 3.1415))

# Combine the learning rate schedules
def phased_lr_lambda(step, total_updates_phase1=1000, total_updates_phase2=299000):
    if step < total_updates_phase1:
        return lr_lambda_phase1(step, total_updates_phase1=total_updates_phase1)
    else:
        return lr_lambda_phase2(step - total_updates_phase1, total_updates_phase2=total_updates_phase2)
    

# https://arxiv.org/pdf/2312.03876.pdf
def lr_lambda_phase1(epoch, num_epochs=100, warmup_epochs=10):

    total_epochs = num_epochs - warmup_epochs
    
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / total_epochs
        return 0.5 * (1 + math.cos(math.pi * progress))