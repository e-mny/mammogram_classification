import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        
        self.densenet_model = models.densenet121(pretrained=False)

        # ## TODO: See if pre-training network affects performance
        # # Load the pre-trained DenseNet model
        # self.densenet_model = models.densenet121(pretrained=True)
        
        # # Freeze the pre-trained layers if desired
        # for param in self.densenet_model.parameters():
        #     param.requires_grad = False
        

        # Get the number of input features to the classifier
        num_features = self.densenet_model.classifier.in_features
        
        # Define custom classifier
        self.custom_classifier = nn.Sequential(
            # nn.Linear(num_features, 512),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, num_classes)

            nn.Linear(num_features, num_classes)
        )
        
        # Replace the classifier in the model with custom classifier
        self.densenet_model.classifier = self.custom_classifier
    
    def forward(self, x):
        return self.densenet_model(x)

# Instantiate the DenseNet class
num_classes = 3  # Change this to the number of classes in your task
model = DenseNet(num_classes)

# Print the model architecture
print(model)