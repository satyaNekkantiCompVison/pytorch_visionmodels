from torch_lr_finder import LRFinder


def find_lr(model, train_loader, optimizer, criterion, device):
    """
    Find best LR.
    """
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader,step_mode="linear",end_lr=1,num_iter=200)
    max_lr = lr_finder.plot(suggest_lr=True, skip_start=0, skip_end=0)
    lr_finder.reset()

    return max_lr[-1]