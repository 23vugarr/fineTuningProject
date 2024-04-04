import torch
import torch.nn as nn
import torchvision.models as model


class ResNet18Based(nn.Module):
    
    def __init__(self):
        super().__init__()
                
        self.resnet18 = model.resnet18(pretrained=True)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))    
        
        self.classifier = torch.nn.Linear(512, 14) 


    def forward(self, image):
        # Get predictions from ResNet18
        resnet_pred = self.resnet18(image).squeeze()
        out = self.classifier(resnet_pred)
        
        return out

class VGG16Based(nn.Module):
    
    def __init__(self):
        super().__init__()
                
        self.vgg16 = model.vgg16(pretrained=True)
        self.vgg16 = torch.nn.Sequential(*(list(self.vgg16.features.children())[:-1]))    
        
        self.classifier = torch.nn.Linear(512 * 16 * 16, 14) 


    def forward(self, image):
        
        vgg_features = self.vgg16(image)
        vgg_features = vgg_features.view(image.size(0), -1)
        
        out = self.classifier(vgg_features)
        
        return out