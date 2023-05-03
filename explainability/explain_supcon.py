"""
Not working as of yet...
    - Issue with gradients not being retained?

-- Ryan
"""

import torch 
import torch.nn as nn 
from captum.attr import IntegratedGradients
from tqdm.notebook import tqdm

class LabelExplainer(nn.Module):
    def __init__(self, model, criterion, ref_img, ref_label, device = 'cpu'):
        """
        Unlike cross-entropy, we calculate supervised contrastive loss based on batches.
        The goal is to get image explanations for a given label.

        Using ideas from contrastive-learning interpretability:
            https://github.com/HealthML/ContIG/blob/main/feature_explanations.py

        Parameters:
            model -- Image encoder
            ref_img -- Reference image batch
            ref_label -- Reference label batch
        """
        super().__init__()
        self.model = model 
        self.model.eval()
        
        self.ref_label = ref_label
        with torch.no_grad():
            self.ref_img_embedding = self.model(ref_img)

        self.criterion = criterion
    
    # def set_img_embedding(self, img):
    #     with torch.no_grad():
    #         self.img_embedding = (
    #             self.model(img.unsqueeze(0))
    #         )
    
    def forward(self, img, label):
        """

        Parameters
            img -- (batch_size, C, H, W)
            label -- (batch_size, )
        """
        loss = torch.zeros(img.shape[0])
        for i in range(img.shape[0]):
            img_embedding_cat = torch.cat(
                [
                    self.ref_img_embedding,
                    self.model(img[i, ...].unsqueeze(0))
                ]
            )
            label_cat = torch.cat(
                [
                    self.ref_label,
                    label[i].unsqueeze(-1)
                ]
            )
            loss[i] = self.criterion(img_embedding_cat, label_cat)
        return loss
    

def explain_image(img, label, explainer, attr_kwargs = dict(), device = 'cpu', use_tqdm = True):
    gen = tqdm(range(len(img))) if use_tqdm else range(len(img)) 

    attr_tests = []
    # for each observation
    for i in gen: 
        # explainer.set_img_embedding(img[i].unsqueeze(0).to(device)) 
        attr_test = IntegratedGradients(explainer).attribute(
            (img, label), **attr_kwargs,
        )
        attr_tests.append(attr_test.detach().cpu().double().numpy())
    return attr_tests
