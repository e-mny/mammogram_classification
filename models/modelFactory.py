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
        elif self.model_name == "xception":
            return self.create_xception()
        else:
            raise ValueError("Unsupported model name.")

    def create_resnet18(self):
        model = models.resnet18(pretrained=self.pretrained)
        for name, param in model.named_parameters():
            param.requires_grad = False

        for name, param in model.layer4[-1:].named_parameters():
            # if "2" in name: # Second last layer of layer4[-1]
            #     param.requires_grad = True
            if "2" in name: # Last layer of layer4[-1]
                param.requires_grad = True
                
        # num_features = model.fc.in_features
        # model.fc = nn.Linear(num_features, self.num_classes)
        model = replaceFCLayer(model, self.num_classes)
        return model
    
    def create_resnet34(self):
        model = models.resnet34(pretrained=self.pretrained)
        for name, param in model.named_parameters():
            param.requires_grad = False

        for name, param in model.layer4[-1:].named_parameters():
            if "2" in name: # last layer of layer4[-1]
                param.requires_grad = True
            # if "3" in name: # Last layer of layer4[-1]
            #     param.requires_grad = True
        
        # num_features = model.fc.in_features
        # model.fc = nn.Linear(num_features, self.num_classes)
        model = replaceFCLayer(model, self.num_classes)

        return model
    
    def create_resnet50(self):
        model = models.resnet50(pretrained=self.pretrained)
        
        # Loading pretrained weights from MedMNIST
        # WEIGHTS_PATH = "/home/emok/sq58/Code/base_mammo/models/pneumoniamnist/resnet50_224_1.pth"
        # checkpoint = torch.load(WEIGHTS_PATH)
        
        # # To load checkpoints with correct num of output classes properly
        # if "chestmnist" in WEIGHTS_PATH:
        #     num_classes = 14
        # elif "pneumoniamnist" in WEIGHTS_PATH:
        #     num_classes = 2
        # model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

        # model.load_state_dict(checkpoint['net'])

        for name, param in model.named_parameters():
            # if "bn" in name: # Following EMBED Screening Model Paper
            #     param.requires_grad = True
            # else:
            param.requires_grad = False
            
        for name, param in model.layer4[-1:].named_parameters():
            # param.requires_grad = True # All layers
            # if "2" in name: # Second last layer of layer4[-1]
            #     param.requires_grad = True
            if "3" in name: # Last layer of layer4[-1]
                param.requires_grad = True
            
        # for param in model.layer3[-1:].parameters():
        #     param.requires_grad = True

        # num_features = model.fc.in_features
        # model.fc = nn.Linear(num_features, self.num_classes)
        model = replaceFCLayer(model, self.num_classes)
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
    
    def create_densenet121(self):
        model = models.densenet121(pretrained=self.pretrained)
        model.features.conv0 = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        for name, param in model.named_parameters():
            param.requires_grad = False

        model = replaceFCLayer(model, self.num_classes)

        # Unfreeze the very last convolutional layer
        for param in model.features.denseblock4.denselayer16.parameters():
            param.requires_grad = True

        return model

    def create_efficientnet_b0(self):
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._conv_stem = nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        for name, param in model.named_parameters():
            param.requires_grad = False

        model = replaceFCLayer(model, self.num_classes)

        # Unfreeze the very last convolutional layer
        for param in model._conv_head.parameters():
            param.requires_grad = True

        return model
    
    def create_resnext50_32x4d(self):
        model = models.resnext50_32x4d(pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = False

        model = replaceFCLayer(model, self.num_classes)

        # Unfreeze the very last convolutional layer
        for param in model.layer4[-1:].parameters():
            param.requires_grad = True

        return model

    def create_mobilenet_v2(self):
        model = models.mobilenet_v2(pretrained=self.pretrained)
        model.features[0][0] = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Modify the final fully connected layer for custom number of classes
        model = replaceFCLayer(model, self.num_classes)

        # Unfreeze the very last convolutional layer
        for param in model.features[-1:].parameters():
            param.requires_grad = True

        return model

    def create_xception(self):
        model = timm.create_model("xception", pretrained=self.pretrained, in_chans=self.input_channels)
        for name, param in model.named_parameters():
            param.requires_grad = False

        model = replaceFCLayer(model, self.num_classes)

        # Unfreeze the very last convolutional layer
        for param in model.conv4.parameters():
            param.requires_grad = True

        return model
    
    def create_efficientnet_b7(self):
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=self.num_classes)
        # Replace the first convolutional layer to accept single-channel input
        model._conv_stem = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        for param in model.parameters():
            param.requires_grad = False
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

def reset_weights(model):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def replaceFCLayer(model, out_classes):
    # Find the last fully connected layer
    # final_layer_name = None
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Linear):
    #         final_layer_name = name

    # if final_layer_name is not None:
    #     # Replace the final layer
    #     num_features = getattr(model, final_layer_name).in_features
    
    num_features = model.fc.in_features
    
    # From paper
    classifier_layer = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    
    # setattr(model, final_layer_name, classifier_layer)
    model.fc = classifier_layer

    
    return model

def create_model(model_name, num_classes, input_channels, pretrained):
    """Create a model factory object."""
    return ModelFactory(model_name=model_name, num_classes=num_classes, input_channels=input_channels, pretrained=pretrained).create_model()
