import torch

def l1_regularization(model, lambda_l1=0.001):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def LOSS(loss_func, outputs, masks):
    total_loss=0
    for i in loss_func:
        total_loss+=i(outputs,torch.argmax(masks,dim=1))
    return total_loss