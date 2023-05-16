# general models here
import torch 
import torch.nn as nn
from torchvision.models import resnet50

class ResNetFeat(nn.Module):
    """
    resnet feature extractor
    """
    def __init__(self, num_channels, num_classes, out_dim):
        super(ResNetFeat, self).__init__()
        self.resnet = resnet50(weights = None)#'IMAGENET1K_V2') # finetuning
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, 7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes) 
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        # might need to add a projection head and throw it away afterwards... as per supcon paper
        self.proj = nn.Sequential(
            nn.Linear(2048, 2048), 
            nn.ReLU(inplace=  True),
            nn.Linear(2048, out_dim)
        )

    def forward(self, x):
        z = self.feature_extractor(x)
        z = z.squeeze((-2, -1))
        z = self.proj(z)
        z = F.normalize(z, dim = 1)
        return z#.unsqueeze(1) # (bsz, num_feat)

    # def forward(self, x):
    #     return self.resnet(x)

class Squeeze(nn.Module): 
    def forward(self, x):
        return x.squeeze((-2, -1))
    
class ResNet50(nn.Module):
  def __init__(self, num_classes):
    # super class 
    super(ResNet50, self).__init__()
    self.resnet = resnet50(pretrained=False) # set with pretrained for now 

    # remove last layer: (fc): Linear(in_features=2048, out_features=1000, bias=True)
    self.features = nn.Sequential(*list(self.resnet.children())[:-1])

    # add layers 
    #model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.fc = nn.Sequential(
        nn.Linear(in_features=self.resnet.fc.in_features, out_features=1000), 
        nn.BatchNorm1d(1000), 
        nn.Dropout(0.2), 
        nn.Linear(1000, 2)
    )
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x