import torch 
import copy
import numpy as np
from adversarial import pgd_linf

from tqdm import tqdm

VAL_TYPE = 'standard'

def epoch_standard(model, criterion, loader, epoch, optimizer = None, device = 'cpu', show_acc = False, multi_view = False):
    """
    standard epoch
    """
    if optimizer:
        model.train()
        mode = 'Train'
    else:
        model.eval()
        mode = 'Val'

    train_loss = []
    batches = tqdm(enumerate(loader), total=len(loader))
    batches.set_description("Epoch NA: Loss (NA)")
    correct = 0
    count = 0
    for batch_idx, (x, y, meta) in batches:
        x, y = [x[0].to(device), x[1].to(device)], y.to(device)
        
        if multi_view: # concatenate views
            z = torch.cat([model(x[0]).unsqueeze(1), 
                           model(x[1]).unsqueeze(1)], dim = 1)
            #print(z.shape)
        else:
            z = model(x[0])
        loss = criterion(z, y)
            
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(loss.item())

        if show_acc:
            correct += (z.max(axis = 1).indices == y).float().sum()
            count += y.shape[0]

        train_loss.append(loss.item())

        acc = 100 * correct / count if show_acc else torch.tensor([-1.0])

        batches.set_description(
            "Epoch {:d}: {:s} Loss ({:.2e}) ACC ({:.2e})".format(
                epoch, mode, loss.item(), acc.item()
            )
        )

    return np.mean(train_loss), acc.detach().cpu().numpy()#(100 * correct/count).detach().cpu().numpy()

def epoch_adversarial(model, criterion, loader, epoch, eps=0.2, step_size=1e-2, adv_steps=40, optimizer = None, device = 'cpu', show_acc = False, multi_view = False):
    """
    eps -- l_inf bound
    step_size -- delta stepsize for inner maximization
    adv_steps -- number of steps of adversarial pertubation
    """
    if optimizer:
        model.train()
        mode = 'Train'
    else:
        model.eval()
        mode = 'Val'

    train_loss = []
    batches = tqdm(enumerate(loader), total=len(loader))
    batches.set_description("Epoch NA: Adversarial Loss (NA)")
    correct = 0
    count = 0
    for batch_idx, (x, y, meta) in batches:
        x, y = [x[0].to(device), x[1].to(device)], y.to(device)
        # inner maximization
        delta = pgd_linf(model, x, y, criterion, eps, step_size, adv_steps, multi_view = multi_view)

        # outer minimization
        if multi_view: # concatenate views
            z = torch.cat([model(x[0] + delta[0]).unsqueeze(1), 
                           model(x[1] + delta[1]).unsqueeze(1)], dim = 1)
        else:
            z = model(x[0] + delta[0])
        loss = criterion(z, y)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(loss.item())

        if show_acc:
            correct += (z.max(axis = 1).indices == y).float().sum()
            count += y.shape[0]
        acc = 100 * correct / count if show_acc else torch.tensor([-1.0])

        batches.set_description(
            "Adversarial Epoch {:d}: {:s} Loss ({:.2e}) ACC ({:.2e})".format(
                epoch, mode, loss.item(), acc.item()
            )
        )
    return np.mean(train_loss), acc.detach().cpu().numpy()

def fit_model(net, criterion, train_loader, val_loader, optimizer, n_epochs, adv_kwargs = dict(), device = 'cpu', use_adv = True, show_acc = False, use_val_acc = False, multi_view = False, scheduler = None):
    ###### Train Model ######
    train_losses = {'standard':[], 'adversarial':[]}
    val_losses = {'standard':[], 'adversarial':[]}
    best_val = float("inf")
    best_epoch = 0
    best_acc = 0
    train_accs = []
    val_accs = []

    for epoch in tqdm(range(n_epochs)):

        # train 
        train_loss, train_acc = epoch_standard(net, criterion, train_loader, epoch, optimizer, device = device, show_acc = show_acc, multi_view = multi_view)
        train_losses['standard'].append(train_loss)
        if show_acc:
            train_accs.append(train_acc)
        if use_adv:
            train_loss_adv, _ = epoch_adversarial(net, criterion, train_loader, epoch, **adv_kwargs, optimizer = optimizer, device = device, show_acc = show_acc, multi_view = multi_view)
            train_losses['adversarial'].append(train_loss_adv)

        # eval 
        val_loss, val_acc = epoch_standard(net, criterion, val_loader, epoch, optimizer = None, device = device, show_acc = show_acc, multi_view = multi_view)
        val_losses['standard'].append(val_loss)
        if show_acc:
            val_accs.append(val_acc)
        # if use_adv:
        #     val_loss_adv, _ = epoch_adversarial(net, criterion, val_loader, epoch, **adv_kwargs, optimizer = None, device = device, show_acc = show_acc)
        #     val_losses['adversarial'].append(val_loss_adv)
        
        if scheduler is not None:
            scheduler.step()

        # retain best val
        if not use_val_acc:
            if best_val >= val_losses[VAL_TYPE][-1]:
                best_val = val_losses[VAL_TYPE][-1]
                best_epoch = epoch
                print(f"Updating at {best_epoch}")
                # save model parameter/state dictionary
                best_model = copy.deepcopy(net.state_dict())
        else:
            if best_acc <= val_accs[-1]:
                best_acc = val_accs[-1]
                best_epoch = epoch
                print(f"Updating at {best_epoch}")
                # save model parameter/state dictionary
                best_model = copy.deepcopy(net.state_dict())

    # load best weights
    print(f"Best epoch at {best_epoch} with {VAL_TYPE} loss: {best_val}")
    net.load_state_dict(best_model)
    return train_losses, val_losses, train_accs, val_accs
