from dicom_conversion import load_and_preprocess_dicom
from normalize_intensity import normalize_intensity
from resample import resample_to_resolution
import matplotlib.pyplot as plt


dicom_path = '/Users/enoch/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/CNYang/OFYP/Coding/Data/CBIS-DDSM/1.3.6.1.4.1.9590.100.1.2.122706985411446472401146362143620569675/1-1.dcm'
pixel_data, dicom_data = load_and_preprocess_dicom(dicom_path)

normalized_data = normalize_intensity(pixel_data)

new_resolution = (224, 224)
resampled_image = resample_to_resolution(normalized_data, dicom_data, new_resolution)

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(resampled_image, cmap='gray')
axs[1].imshow(pixel_data, cmap='gray')
# plt.show()

print((resampled_image).shape)

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])