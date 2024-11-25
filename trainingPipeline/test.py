import torch
import sys
sys.path.append('../')
from utils import LOSS,EVAL_SCORES,l1_regularization
def testing(model, dataset_loader, num_classes,device,loss_fn):
    model.eval()
    loss = 0.0
    iou = 0.0
    f1=0.0
    with torch.inference_mode():
        for batch_idx, (images, masks) in enumerate(dataset_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = outputs.contiguous()
            loss_temp=LOSS(loss_fn,outputs,masks) +l1_regularization(model)
            loss += loss_temp.item()

            per_batch_iou,per_batch_f1 = EVAL_SCORES(outputs, masks, num_classes)
            iou += per_batch_iou
            f1 += per_batch_f1


        loss /= len(dataset_loader)
        iou /= len(dataset_loader)
        f1 /= len(dataset_loader)
        return loss, iou, f1