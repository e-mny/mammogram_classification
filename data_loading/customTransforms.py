from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms

# Define a custom transformation that includes CLAHE
class CLAHETransform:
    def __init__(self, clip_limit=1.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        # Convert the PyTorch tensor to a NumPy array
        image = np.array(image)

        # Convert the image to grayscale (if it's not already)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe_image = clahe.apply(image)

        # Convert the image back to a PyTorch tensor
        clahe_image = Image.fromarray(clahe_image)

        return clahe_image
    

class CenterCropWithMainContent(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        width, height = image.size
        target_width, target_height = self.output_size

        # Calculate the cropping box
        left = max(0, (width - target_width) // 2)
        upper = max(0, (height - target_height) // 2)
        right = min(width, left + target_width)
        lower = min(height, upper + target_height)

        # Crop the image
        cropped_image = transforms.functional.crop(image, upper, left, target_height, target_width)

        return cropped_image