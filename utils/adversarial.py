import torch

def pgd_linf(model, x, y, criterion, eps, step_size, adv_steps, randomize = True, multi_view = False):
    """ 
    PGD l-inf norm
    https://adversarial-ml-tutorial.org/adversarial_examples/
    x = [x0, x1] where x0 is view 1, x1 is view 2
    """
    if randomize: 
        delta1 = torch.rand_like(x[0], requires_grad=True) # uniform random -> can start pertubations at different locations
        delta1.data = delta1.data * 2 * eps - eps
        delta2 = torch.rand_like(x[1], requires_grad=True) # uniform random -> can start pertubations at different locations
        delta2.data = delta2.data * 2 * eps - eps
    else:
        delta1 = torch.zeros_like(x[0], requires_grad=True) # start w/ no noise
        delta2 = torch.zeros_like(x[1], requires_grad = True)
        
    for t in range(adv_steps):
        if multi_view:
            z1 = model(x[0] + delta1).unsqueeze(1)
            z2 = model(x[1] + delta2).unsqueeze(1)
            z = torch.cat([z1, z2], dim = 1)
        else:
            z = model(x[0] + delta1)
        #print(z.shape)
        loss = criterion(z, y) # perturbed loss
        loss.backward() # gradiets wrt delta 
        delta1.data = (delta1 + step_size*delta1.grad.detach().sign()).clamp(-eps,eps) # linf clamp
        if multi_view:
            delta2.data = (delta2 + step_size*delta2.grad.detach().sign()).clamp(-eps,eps) # linf clamp
            delta2.grad.zero_()
        delta1.grad.zero_() # reset grads
    return [delta1.detach(), delta2.detach()] # optimal perturbation projected onto linf ball
