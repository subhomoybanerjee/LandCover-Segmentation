import wandb
from .test import testing
import sys
sys.path.append('../')
from utils import LOSS,EVAL_SCORES,l1_regularization
def fit(model, train_loader, val_loader, device, loss_fn,
        num_epochs,early_stopping,optimizer,scheduler,wb,num_classes):
    train_losses, val_losses = [], []
    train_iou_scores, val_iou_scores = [], []
    train_f1_scores, val_f1_scores = [], []
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_f1 = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            outputs = outputs.contiguous()

            loss = LOSS(loss_fn,outputs,masks) + l1_regularization(model)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            per_batch_iou,per_batch_f1 = EVAL_SCORES(outputs, masks, num_classes)
            train_iou += per_batch_iou
            train_f1 += per_batch_f1

            sys.stdout.write(f"\rEpoch {epoch + 1}/{num_epochs}, "
                             f"Batch {batch_idx + 1}, "
                             f"Loss: {train_loss / (batch_idx + 1):.4f}, "
                             f"IoU: {train_iou / (batch_idx + 1):.4f}, "
                             f"F1: {train_f1 / (batch_idx + 1):.4f}")
            sys.stdout.flush()

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_f1 /= len(train_loader)

        val_loss, val_iou, val_f1 = testing(model, val_loader, num_classes, device, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_iou_scores.append(train_iou)
        val_iou_scores.append(val_iou)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        sys.stdout.write('\r---------------------------------------------------------------------')
        print(f'\nEpoch {epoch + 1}/{num_epochs}'
              f'\nTrain Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train F1: {train_f1:.4f} '
              f'\nVal Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val F1: {val_f1:.4f}\n')


        scheduler.step(val_loss)

        if wb == True:
            wandb.log({"train iou": train_iou, "train_f1":train_f1,"train loss": train_loss,
                       "val iou": val_iou,"val f1":val_f1 ,"val loss": val_loss})

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_iou_scores,train_losses,train_f1_scores,val_iou_scores,val_losses,val_f1_scores