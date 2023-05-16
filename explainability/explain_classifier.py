from captum.attr import IntegratedGradients
from utils.viz import visualize_image_attr
from matplotlib.colors import LinearSegmentedColormap
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    #print(img.unsqueeze(1).shape)
    attributes = ig.attribute(img.to(device), 
                               target=target.to(device), 
                               **attr_kwargs)
    return attributes

# Captum.ai IG plotting functions
def visualize_classifier_exp(attr, img, plt_fig_axis):
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    _ = visualize_image_attr(attr.squeeze().cpu().detach().numpy(), # normalize over one channel/depth
                                img.squeeze().cpu().detach().numpy(),
                                plt_fig_axis = plt_fig_axis,
                                method='heat_map',
                                cmap=default_cmap,
                                show_colorbar=True,
                                sign='positive',
                                outlier_perc=1)