from pytorch_grad_cam import GradCAM
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image

def generateHeatMap(sample_image, sample_title, model, device):
    currdatetime = datetime.now()
    formatted_datetime = currdatetime.strftime("%m-%d-%y-%H%M%S")
    output_dir = os.path.join("./visualization/samples", formatted_datetime)
    os.mkdir(output_dir)
    for i in range(len(sample_image)):
        plt.clf() # Clear figure

        image = sample_image[i]
        title = sample_title[i]
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.to(device)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        # print(image_tensor.shape)
        
        # # Grad-CAM
        # class_idx = predicted_class.item()
        # feature_layer = [model.classifier[-1]]  # Example: last layer of ResNet-50
        # feature_layer = [model.layer4[-1].conv2]  # Example: last layer of ResNet-34
        feature_layer = [model.layer4[-1].conv3]  # Example: last layer of ResNet-50
        for param in feature_layer[0].parameters():
            param.requires_grad = True
        # print(feature_layer)
        cam = GradCAM(model=model, target_layers=feature_layer, use_cuda=True)
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        # print(image_tensor.shape)
        grayscale_cam = cam(input_tensor=image_tensor, targets=None, aug_smooth=True)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True, image_weight=0.8)
        
        # Display the result
        plt.imshow(visualization)
        plt.axis('off')
        plt.title(title)
        combined_output_path = os.path.join(output_dir, f'{title}-{i}.jpg')
        plt.savefig(combined_output_path)
