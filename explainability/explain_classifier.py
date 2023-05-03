from captum.attr import IntegratedGradients

def explain_classifier(model, img, target, attr_kwargs = dict()):
    """
    Return integrated gradients for given prediction.

    Parameters:
        model -- Classifier of interest
        img -- Tensor of size batchsize x C x H x W 
        target -- Target label
        attr_kwargs -- Explainability method kwargs
    """
    ig = IntegratedGradients(model)
    img.requires_grad = True 
    attributes = ig.attribute(img.unsqueeze(1), 
                               target=target, 
                               **attr_kwargs)
    return attributes