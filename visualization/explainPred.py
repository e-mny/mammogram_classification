from pytorch_grad_cam import GradCAM
import torch
from pytorch_grad_cam.utils.image import deprocess_image, show_cam_on_image
import os
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def generateHeatMap(dataloader, model, device):
    model_name = str(model).split("(")[0]

    # if "resnet" in model.name.lower():
    #     target_layer = [model.layer4[-1]]
    # elif "densenet" in model.name.lower():
    target_layer = [model.layer4[-1]]
    inputs, _ = next(iter(dataloader))
    input_pic = np.transpose(inputs.numpy(), (0, 2, 3, 1))
    input_pic = [np.clip(image.astype(np.uint8), 0, 255) for image in input_pic]
    input_pic = [image[:, :, 0] for image in input_pic]
    print(input_pic[0].shape)
    cam = GradCAM(model=model, use_cuda = torch.cuda.is_available(), target_layers = target_layer)
    output_dir = "./visualization/samples"
    currdatetime = datetime.now()
    formatted_datetime = currdatetime.strftime("%d-%m-%y-%H%M%S")
    input_path = os.path.join(output_dir, f'{model_name}_{formatted_datetime}_input.jpg')
    output_path = os.path.join(output_dir, f'{model_name}_{formatted_datetime}_output.jpg')
    combined_output_path = os.path.join(output_dir, f'{model_name}_{formatted_datetime}_combined.jpg')

    
    # model.eval().to(device)
    # inputs = inputs.to(device)
    # outputs = model(inputs)
    batch_size = dataloader.batch_size
    num_rows = 4
    num_cols = int(batch_size / num_rows)

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))

    # Iterate through the batch and plot each image
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        image = input_pic[i]  # Transpose channels for display
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(input_path)
    
    input_pic = input_pic.to(device)
    cam_pics = cam(input_tensor = input_pic, targets=None, eigen_smooth = True)
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        cam = cam_pics[i]  # Transpose channels for display
        axes[row, col].imshow(cam)
        axes[row, col].axis('off')
    plt.tight_layout()
    combined = show_cam_on_image(input_pic, cam_pics)
    
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        combined_pic = combined[i]  # Transpose channels for display
        axes[row, col].imshow(combined_pic)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(combined_output_path)

        
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp

def gradCamFunc(dataloader, model, device):
    model_name = str(model).split("(")[0]
    output_dir = "./visualization/samples"
    currdatetime = datetime.now()
    formatted_datetime = currdatetime.strftime("%d-%m-%y-%H%M%S")

    input_path = os.path.join(output_dir, f'{model_name}_{formatted_datetime}_input.jpg')
    edgemask_path = os.path.join(output_dir, f'{model_name}_{formatted_datetime}_edge_mask.jpg')
    histogram_path = os.path.join(output_dir, f'{model_name}_{formatted_datetime}_histogram.jpg')
    combined_output_path = os.path.join(output_dir, f'{model_name}_{formatted_datetime}_combined.jpg')

    model.eval()
    inputs, _ = next(iter(dataloader))
    input_pic = np.transpose(inputs.numpy(), (0, 2, 3, 1))
    input_pic = [np.clip(image.astype(np.uint8), 0, 255) for image in input_pic]
    print(input_pic[0].shape)
    # first_pic, _ = inputs[0]
    
    batch_size = dataloader.batch_size
    num_rows = 4
    num_cols = int(batch_size / num_rows)

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))

    # Iterate through the batch and plot each image
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        # image = input_pic[i].permute(1, 2, 0).numpy()  # Transpose channels for display
        image = input_pic[i]  # Transpose channels for display
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(input_path)
    
    # Histogram equalization
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        # image = input_pic[i].permute(1, 2, 0).numpy()  # Transpose channels for display
        image = input_pic[i]  # Transpose channels for display
        # Compute the histogram of the image
        hist, bins = np.histogram(image, bins=256, range=(0, 256))

        # Calculate the cumulative distribution function (CDF)
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        # Apply histogram equalization
        equalized_image = np.interp(image, bins[:-1], cdf_normalized)
        axes[row, col].imshow(equalized_image)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(histogram_path)
    
    # Edge Mask
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        # image = input_pic[i].permute(1, 2, 0).numpy()  # Transpose channels for display
        image = input_pic[i] # Transpose channels for display
        edges = cv2.Canny(image, threshold1 = 100, threshold2 = 200)
        axes[row, col].imshow(edges)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(edgemask_path)

    cam_extractor = SmoothGradCAMpp(model)
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        # curr_pic = input_pic[i].permute(1, 2, 0).numpy()  # Transpose channels for display
        curr_pic = input_pic[i]  # Transpose channels for display
        # with SmoothGradCAMpp(model) as cam_extractor:
        # Preprocess your data and feed it to the model
        out = model(curr_pic)
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        # Resize the CAM and overlay it
        result = overlay_mask(to_pil_image(curr_pic), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

        axes[row, col].imshow(result)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(combined_output_path)

    # combined = show_cam_on_image(input_pics, cam_pics)
    # for i in range(batch_size):
    #     row = i // num_cols
    #     col = i % num_cols
    #     combined_pic = combined[i]  # Transpose channels for display
    #     axes[row, col].imshow(image)
    #     axes[row, col].axis('off')
    # plt.tight_layout()
    # plt.savefig(combined_output_path)
    