
from utils.train import train
from utils.test import test

def fit_model(model, device,epochs, train_loader, test_loader, optimizer, criterion, l1_factor, scheduler, scheduler_params):
    """
    Train and evaluate for given epochs.
    """
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    lr_trend = []

    if scheduler == "OneCycleLR":
        from torch.optim.lr_scheduler import OneCycleLR 
        max_epoch =  scheduler_params["max_epoch"]
        if max_epoch is None:
            pct_start = 0.3
        else:
            pct_start =max_epoch/epochs

        scheduler = OneCycleLR(optimizer,max_lr=scheduler_params["max_lr"],epochs=epochs,steps_per_epoch=len(train_loader), div_factor=scheduler_params["div_factor"],pct_start=pct_start,
                                anneal_strategy=scheduler_params["anneal_strategy"],three_phase = scheduler_params["three_phase"])

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}:")
        train(model,device,train_loader,optimizer,l1_factor,criterion,lr_trend,scheduler,0.1,train_accuracy,train_losses)
        test(model, device, test_loader, criterion, test_accuracy, test_losses)

    return train_accuracy, train_losses, test_accuracy, test_losses
