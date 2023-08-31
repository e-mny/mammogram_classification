import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import timm

class ModelFactory:
    def __init__(self, model_name, num_classes, input_channels=3, pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.pretrained = pretrained
        
    def create_model(self):
        if self.model_name == "resnet18":
            return self.create_resnet18()
        elif self.model_name == "resnet34":
            return self.create_resnet34()
        elif self.model_name == "resnet50":
            return self.create_resnet50()
        elif self.model_name == "vgg16":
            return self.create_vgg16()
        elif self.model_name == "vgg19":
            return self.create_vgg19()
        elif self.model_name == "alexnet":
            return self.create_alexnet()
        elif self.model_name == "googlenet":
            return self.create_googlenet()
        elif self.model_name == "inception_v3":
            return self.create_inception_v3()
        elif self.model_name == "resnext50_32x4d":
            return self.create_resnext50_32x4d()
        elif self.model_name == "densenet121":
            return self.create_densenet121()
        elif self.model_name == "efficientnet_b0":
            return self.create_efficientnet_b0()
        elif self.model_name == "efficientnet_b7":
            return self.create_efficientnet_b7()
        elif self.model_name == "mobilenet_v2":
            return self.create_mobilenet_v2()
        elif self.model_name == "mobilenet_v3_small":
            return self.create_mobilenet_v3_small()
        elif self.model_name == "mobilenet_v3_large":
            return self.create_mobilenet_v3_large()
        elif self.model_name == "xception":
            return self.create_xception()
        # Add more model creation methods here...
        else:
            raise ValueError("Unsupported model name.")

    def create_resnet18(self):
        model = models.resnet18(pretrained=self.pretrained)
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_resnet34(self):
        model = models.resnet34(pretrained=self.pretrained)
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_resnet50(self):
        model = models.resnet50(pretrained=self.pretrained)
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_vgg16(self):
        model = models.vgg16(pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_vgg19(self):
        model = models.vgg19(pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_alexnet(self):
        model = models.alexnet(pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_googlenet(self):
        model = models.googlenet(pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_inception_v3(self):
        model = models.inception_v3(pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_resnext50_32x4d(self):
        model = models.resnext50_32x4d(pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False
            
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_wide_resnet50_2(self):
        model = models.wide_resnet50_2(pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False
            
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model

    def create_densenet121(self):
        model = models.densenet121(pretrained=self.pretrained)
        model.features.conv0 = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False) # To adjust input size
        for name, param in model.named_parameters():
            if "classifier" in name:
                # param.requires_grad = True
                break
            else:
                param.requires_grad = False
            
        
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, self.num_classes)
        return model

    def create_efficientnet_b0(self):
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=self.num_classes)
        # Replace the first convolutional layer to accept single-channel input
        model._conv_stem = nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        for name, param in model.named_parameters():
            if "_fc" in name:
                # param.requires_grad = True
                break
            else:
                param.requires_grad = False
                
        # print(dir(model))
        for param in model._blocks[-3:]:
            param.requires_grad = True
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, self.num_classes)
        return model
    
    def create_efficientnet_b7(self):
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=self.num_classes)
        # Replace the first convolutional layer to accept single-channel input
        model._conv_stem = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def create_mobilenet_v2(self):
        model = models.mobilenet_v2(pretrained=self.pretrained)
        model.features[0][0] = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        # for param in model.features[-3:].parameters(): # Last 3 layers
        #     print(param)
        #     param.requires_grad = True
        # Modify the final fully connected layer for custom number of classes
        # num_features = model.classifier[1].in_features
        # model.classifier[1] = nn.Linear(num_features, self.num_classes)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=2, bias=True)
        )
        return model

    def create_mobilenet_v3_small(self):
        model = models.mobilenet_v3_small(pretrained=self.pretrained)
        # model.features[0][0] = nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        for param in model.parameters():
            param.requires_grad = False
        
        # Modify the final fully connected layer for custom number of classes
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, self.num_classes)
        return model

    def create_mobilenet_v3_large(self):
        model = models.mobilenet_v3_large(pretrained=self.pretrained)
        model.features[0][0] = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        for param in model.parameters():
            param.requires_grad = False
        
        # Modify the final fully connected layer for custom number of classes
        # num_features = model.classifier[1].in_features
        # model.classifier[1] = nn.Linear(num_features, self.num_classes)
        return model
    
    def create_xception(self):
        model = timm.create_model("xception", pretrained=self.pretrained, in_chans=self.input_channels)
        for name, param in model.named_parameters():
            if "fc" in name:
                break
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        return model

def printTrainableParams(model):
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Number of trainable parameters:", num_trainable_params)
    print("Total number of parameters: ", total_params)

def freezeLayers(model):
    for name, param in model.named_parameters():
        if "fc" in name or "classifier" in name:  # Update this condition to match your fc layer's name
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

# # Example usage
# num_classes = 2
# input_channels = 1
# pretrained = True

# # # Create a GoogLeNet model
# model_factory = ModelFactory(model_name="mobilenet_v2", num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)
# mobilenet = model_factory.create_model()

# # Print model architecture
# print(mobilenet)
