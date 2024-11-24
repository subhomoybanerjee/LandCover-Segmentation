import torch
def LOSS(loss_func, outputs, masks):
    total_loss=0
    for i in loss_func:
        total_loss+=i(outputs,torch.argmax(masks,dim=1))
    return total_loss