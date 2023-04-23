import torch 
import numpy as np
from adversarial import pgd_linf

from tqdm.notebook import tqdm

def epoch_standard(model, criterion, loader, epoch, optimizer = None, device = 'cpu'):
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

    for batch_idx, (x, y) in batches:
        x, y = x.to(device), y.to(device)
        # outer minimization
        z = model(x)
        #print(z.shape)
        loss = criterion(z, y)
        #print(loss.item())
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(loss.item())
        batches.set_description(
            "Epoch {:d}: {:s} Loss ({:.2e})".format(
                epoch, mode, loss.item()
            )
        )

    return np.mean(train_loss)

def epoch_adversarial(model, criterion, loader, epoch, eps=0.2, step_size=1e-2, adv_steps=10, optimizer = None, device = 'cpu'):
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

    for batch_idx, (x, y) in batches:
        x, y = x.to(device), y.to(device)
        # inner maximization
        delta = pgd_linf(model, x, y, criterion, eps, step_size, adv_steps)

        # outer minimization
        z = model(x + delta)
        loss = criterion(z, y)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(loss.item())
        batches.set_description(
            "Epoch {:d}: {:s} Adversarial Loss ({:.2e})".format(
                epoch,mode, loss.item()
            )
        )
    return np.mean(train_loss)