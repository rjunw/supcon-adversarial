# general models here
import torch 
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50(nn.Module):
    """
    ResNet50 feature extractor
    """
    def __init__(self, num_channels, num_classes, weights = None):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(weights = weights)#'IMAGENET1K_V2') # finetuning
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, 7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes) 
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        """
        Return penultimate features in form [bsz, n_view, 2048] where n_view = 1
        """
        z = self.feature_extractor(x)
        return z.view(z.shape[0], -1).unsqueeze(1)

    def predict(self, x):
        """
        Return logits in form [bsz, num_classes]
        """
        return self.resnet(x)