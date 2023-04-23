import torch

def pgd_linf(model, x, y, criterion, eps, step_size, adv_steps, randomize = True):
    """ 
    PGD l-inf norm
    https://adversarial-ml-tutorial.org/adversarial_examples/
    """
    if randomize: 
        delta = torch.rand_like(x, requires_grad=True) # uniform random -> can start pertubations at different locations
        delta.data = delta.data * 2 * eps - eps
    else:
        delta = torch.zeros_like(x, requires_grad=True) # start w/ no noise
        
    for t in range(adv_steps):
        z = model(x + delta)
        #print(z.shape)
        loss = criterion(z, y) # perturbed loss
        loss.backward() # gradiets wrt delta 
        delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-eps,eps) # linf clamp
        delta.grad.zero_() # reset grads
    return delta.detach() # optimal perturbation projected onto linf ball
