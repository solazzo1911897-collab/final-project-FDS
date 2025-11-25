import torch
import torch.nn as nn
import timm

class GWClassifier(nn.Module):
    """
    Convolutional Neural Network using EfficientNet-B0.
    Restored to the architecture that showed initial success.
    """
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        
        # Load EfficientNet-B0 (Pretrained on ImageNet)
        # Standard input: 3 Channels (RGB), which maps to our 3 Sensors
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0, 
            global_pool='' 
        )
        
        num_features = self.backbone.num_features
        
        # Head with moderate dropout
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features)
        output = self.head(pooled)
        return output