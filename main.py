import os
import torch
import pickle
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
import wandb
from segmentation_models_pytorch.losses import JaccardLoss,DiceLoss,FocalLoss

from data import create_dataloader, preprocess_data
from models import initialize_model
from utils import EarlyStopping, visualizePreds, check_library_versions,plot_metrics
from trainingPipeline import fit, testing


def main():
    check_library_versions()

    splitImages='datasets/split'

    images='images'
    masks='masks'

    train_img_dir = os.path.join(splitImages, f'train/{images}/')
    train_mask_dir = os.path.join(splitImages, f'train/{masks}/')
    val_img_dir = os.path.join(splitImages, f'val/{images}/')
    val_mask_dir = os.path.join(splitImages, f'val/{masks}/')
    test_img_dir = os.path.join(splitImages, f'test/{images}/')
    test_mask_dir = os.path.join(splitImages, f'test/{masks}/')

    normalPath = os.path.join('datasets','normalized','gNormalized')

    patches='landCover.pkl'
    with open(os.path.join(normalPath, patches), 'rb') as file:
        patches = pickle.load(file)
    print(f'\nNormalized Patches loaded')

    device='cpu'
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device='cuda'
    print(f'Device using: {device}')

    # BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # BANDS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    BANDS=[1,2,3]
    num_classes = 5
    batch_size = 8
    print(f'Batch size: {batch_size}, num_classes: {num_classes}, BANDS: {BANDS}')

    train_loader = create_dataloader(
        img_dir=train_img_dir, mask_dir=train_mask_dir,batch_size=batch_size,
        num_classes=num_classes,BANDS=BANDS, patches=patches,
        preprocess_data=preprocess_data,shuffle=True,auggy=True
    )
    print(f'Training dataset loaded')

    val_loader = create_dataloader(
        img_dir=val_img_dir, mask_dir=val_mask_dir,batch_size=batch_size,
        num_classes=num_classes,BANDS=BANDS, patches=patches,
        preprocess_data=preprocess_data,shuffle=False,auggy=False
    )
    print(f'Validation dataset loaded')

    test_loader = create_dataloader(
        img_dir=test_img_dir, mask_dir=test_mask_dir,batch_size=batch_size,
        num_classes=num_classes,BANDS=BANDS, patches=patches,
        preprocess_data=preprocess_data,shuffle=False,auggy=False
    )
    print(f'Testing dataset loaded')

    encoder_name = 'resnet34'
    print(f'Loading encoder: {encoder_name}')

    model=initialize_model(encoder_name,len(BANDS),num_classes,None,device)
    print(f'Architecture loaded: {model._get_name()}')

    wb = False
    if wb == True:
        wandb.init(
            project="U-net++",
            config={
                'batch_size': batch_size,
                'encoder_name': encoder_name,
                'bands': BANDS,
                'normalization': normalPath
            }
        )

    loss_fn_Dice = DiceLoss(mode='multiclass', eps=1e-7, from_logits=True)
    loss_fn_Jaccard = JaccardLoss(mode='multiclass', eps=1e-7, from_logits=True)
    loss_fn_Focal = FocalLoss(mode='multiclass')


    loss_fn = [loss_fn_Dice, loss_fn_Jaccard, loss_fn_Focal]
    print(f'Loss functions loaded: {[i._get_name() for i in loss_fn]}')

    optimizer = Adam(model.parameters(), lr=1e-3)
    print(f'Optimizers loaded: {optimizer.__class__.__name__}')

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True, min_lr=1e-6)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    print(f'Scheduler loaded: {scheduler.__class__.__name__}')

    early_stopping = EarlyStopping(patience=18, verbose=True)
    print(f'Early stopping initialized with patience:{early_stopping.patience}')

    num_epochs = 200
    print(f'number of epochs: {num_epochs}')

    print('\nStarting Loop...')
    train_iou_scores,train_losses,train_f1_scores,val_iou_scores,val_losses,val_f1_scores=(
        fit(model, train_loader, val_loader, device, loss_fn, num_epochs,
        early_stopping, optimizer, scheduler, wb, num_classes))

    results='results'
    output_path=f'{results}/Training_curves'
    plot_metrics(train_losses, val_losses, 'Loss', output_path)
    plot_metrics(train_iou_scores, val_iou_scores, 'IoU Score', output_path)
    plot_metrics(train_f1_scores, val_f1_scores, 'F1 Score',output_path)
    print(f'\ntraining curves saved in {output_path}\n')

    model=initialize_model(encoder_name,len(BANDS),num_classes,'imagenet',device)
    model.load_state_dict(torch.load(f'{results}/checkpoint.pt'))

    trainLoss,trainIou,trainF1=testing(model, train_loader, num_classes, device, loss_fn)
    valLoss,valIou,valF1=testing(model, val_loader, num_classes, device, loss_fn)
    testLoss,testIou,testF1=testing(model, test_loader, num_classes, device, loss_fn)

    with open(os.path.join(results, 'results.txt'), 'w') as f:
        f.write(f'Train Loss: {trainLoss:.4f}, Train IoU: {trainIou:.4f}, Train F1: {trainF1:.4f}\n')
        f.write(f'Val Loss: {valLoss:.4f}, Val IoU: {valIou:.4f}, Val F1: {valF1:.4f}\n')
        f.write(f'Test Loss: {testLoss:.4f}, Test IoU: {testIou:.4f}, Test F1: {testF1:.4f}\n')

    print(f'\n'
          f'trainLoss:{trainLoss}, trainIou:{trainIou}, trainF1:{trainF1}\n'
          f'valLoss:{valLoss}, valIou:{valIou}, valF1:{valF1}\n'
          f'testLoss:{testLoss}, testIou:{testIou}, testF1:{testF1}\n')

    prediction_dir=f'{results}/predicted_images'
    visualizePreds(model, test_loader,device,prediction_dir)

if __name__ == "__main__":
    main()
