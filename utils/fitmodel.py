from utils.train import train
from utils.test import test

def fit_model(net, device, epochs,optimizer, criterion,train_loader, test_loader, use_l1=False, scheduler=None):
    train_accuracy, train_loss_list, test_accuracy, test_loss_value = [],[],[],[]
    lr_trend = []

    for epoch in range(1,epochs+1):
        print("EPOCH: {} (LR: {})".format(epoch, optimizer.param_groups[0]['lr']))
        
        train_acc, train_loss, lr_hist = train(
            model=net, 
            device=device, 
            train_loader=train_loader, 
            criterion=criterion ,
            optimizer=optimizer, 
            use_l1=use_l1, 
            scheduler=scheduler
        )
        test_acc, test_loss = test(net, device, test_loader, criterion)
        # update LR
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)

        train_accuracy.append(train_acc)
        train_loss_list.append(train_loss)
        test_accuracy.append(test_acc)
        test_loss_value.append(test_loss)
        lr_trend.extend(lr_hist)    

    if scheduler:   
        return (train_accuracy, train_loss_list, test_accuracy, test_loss_value, lr_trend)
    else:
        return (train_accuracy, train_loss_list, test_accuracy, test_loss_value)
