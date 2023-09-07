from pytorch_grad_cam import GradCAM
import torch
from pytorch_grad_cam.utils.image import deprocess_image, show_cam_on_image, preprocess_image
import os
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def generateHeatMap(dataloader, model, device):
    model_name = str(model)
    target_layer = None
    if "resnet" in model_name.lower():
        target_layer = [model.layer4[-1]]
    elif "resnext" in model_name.lower():
        target_layer = [model.layer4[-1]]
    elif "densenet" in model_name.lower():
        target_layer = [model.denseblock4[-1]]
    elif "mobilenet" in model_name.lower():
        target_layer = [model.features[-1]]
        model.features[-1].requires_grad = True
    elif "efficient" in model_name.lower():
        target_layer = [model._conv_head]
    elif "xception" in model_name.lower():
        target_layer = [model.conv4]
    elif "mobile" in model_name.lower():
        target_layer = [model.base_model.features[-1]]
        
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    inputs, labels = next(iter(dataloader))
    # input_pic = np.transpose(inputs.numpy(), (2, 3, 0, 1))
    # input_pic = [np.clip(image.astype(np.uint8), 0, 255) for image in inputs.numpy()]
    input_pic = [image[:, :] for image in inputs.numpy()]
    # print(f"First pic shape: {input_pic[0].shape}")
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
        label = labels[i]
        image = np.mean(image, axis=0)
        # print("First Loop")
        # print(image)
        # print(image.shape)
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Label: {label}")
    plt.tight_layout()
    plt.savefig(input_path)
    
    input_pic_tensor = torch.Tensor(np.array(input_pic))
    # input_pic_tensor = np.repeat(input_pic_tensor[:, :, :, np.newaxis], 3, axis = 3)
    input_pic_tensor = input_pic_tensor.to(device)
    # print(input_pic_tensor.shape)
    print(inputs.shape)
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        image = input_pic_tensor[i]
        label = labels[i]
        # print(f"Image: {image.shape}")
        print(f"Input_tensor: {input_pic_tensor.shape}")
        cam_pics = cam(input_tensor = input_pic_tensor, targets=None, eigen_smooth = True)
        cam_img = cam_pics[i]  # Transpose channels for display
        # print(type(cam_img))
        
        axes[row, col].imshow(cam_img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Label: {label}")

    plt.tight_layout()
    plt.savefig(output_path)
    
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        image = input_pic_tensor[i].cpu().numpy()
        # print((np.stack([image] * 3, axis=-1)).shape)
        # image = preprocess_image(np.stack([image] * 3, axis=-1))

        image = input_pic_tensor[i]
        label = labels[i]
        cam_pics = cam(input_tensor = input_pic_tensor, targets=None, eigen_smooth = True)
        cam_img = cam_pics[i]
        # Normalize the heatmap to [0, 1] range
        heatmap_normalized = (cam_img - np.min(cam_img)) / (np.max(cam_img) - np.min(cam_img))
        heatmap_normalized = cv2.resize(heatmap_normalized, (image.shape[1], image.shape[2]))
        # Apply a color map to the normalized heatmap
        heatmap_colormap = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)

        # Overlay the heatmap on the image
        image = np.uint8(image.cpu().numpy() * 255)
        image = np.transpose(image, (1, 2, 0))
        heatmap_colormap = np.uint8(heatmap_colormap)
        # print(f"Image Shape: {image.shape}")
        # print(f"heatmap_colormap Shape: {heatmap_colormap.shape}")
        combined_image = cv2.addWeighted(image, 0.7, np.array(heatmap_colormap), 0.3, 0)
        axes[row, col].imshow(combined_image)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Label: {label}")

        

        
        # cam_img = cam_pics[0, :]
        # print(f"cam_pics shape {cam_pics.shape}")
        # print(f"input_pic_tensor shape {input_pic_tensor.shape}")
        # print(f"cam_img shape {cam_img.shape}")
        # print(f"image shape {image.shape}")
        # # image = np.mean(image, axis=0)
        # # cam_pics = cv2.cvtColor(cam_pics, cv2.COLOR_GRAY2BGR)
        # # input_pic_tensor = cv2.cvtColor(input_pic_tensor, cv2.COLOR_GRAY2BGR)
        # image = image.squeeze(0).permute(1, 2, 0).numpy()[:, :, 0]
        # cam_img = cam_img.permute(1, 2, 0).numpy()[:, :, 0]
        # print(f"cam_img shape {cam_img.shape}")
        # combined = show_cam_on_image(image[:, :], cam_img)
        # # combined_pic = combined[i]  # Transpose channels for display
        # axes[row, col].imshow(combined)
        # axes[row, col].axis('off')
    
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
    