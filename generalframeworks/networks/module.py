import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class Uncertainty_head(nn.Module):   # feature -> log(sigma^2)
    def __init__(self, in_feat=304, out_feat=256):
        super(Uncertainty_head, self).__init__()
        self.fc1 = Parameter(torch.Tensor(out_feat, in_feat))
        self.bn1 = nn.BatchNorm2d(out_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn2 = nn.BatchNorm2d(out_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc1, dim=-1)) # [B, W, H, D]
        x = x.permute(0, 3, 1, 2) # [B, W, H, D] -> [B, D, W, H]
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = F.linear(x, F.normalize(self.fc2, dim=-1))
        x = x.permute(0, 3, 1, 2)
        x = self.bn2(x)
        x = self.gamma * x + self.beta
        x =  torch.log(torch.exp(x) + 1e-6)
        x = torch.sigmoid(x)
        
        return x

class Classifier(nn.Module):
    def __init__(self, in_feat=304, num_classes=21):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, 256, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_feat=256, num_classes=19):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(304, 256, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x_low: torch.Tensor, x: torch.Tensor):
        x_low = self.conv1(x_low)
        x_low = self.bn1(x_low)
        x_low = self.relu1(x_low)
        x = F.interpolate(x, size=x_low.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_low, x], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        return x