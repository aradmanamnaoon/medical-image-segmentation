from pathlib import Path
from monai.transforms import LoadImage
import matplotlib.pyplot as plt


DATA_DIR = Path(r"C:\Users\saba\Downloads\Task02_Heart\Task02_Heart")


images_dir = DATA_DIR / "imagesTr"
labels_dir = DATA_DIR / "labelsTr"


image_files = sorted(images_dir.glob("*.nii.gz"))
label_files = sorted(labels_dir.glob("*.nii.gz"))


# Select one training example
image_path = image_files[0]
label_path = label_files[0]


# Load image and mask
loader = LoadImage(image_only=False)

image, metadata = loader(str(image_path))
label, _ = loader(str(label_path))


# Convert to numpy
image_np = image.numpy()
label_np = label.numpy()


# Middle slice
slice_index = image_np.shape[2] // 2


image_slice = image_np[:, :, slice_index]
label_slice = label_np[:, :, slice_index]


# Visualization

plt.figure(figsize=(12,4))


plt.subplot(1,3,1)
plt.imshow(image_slice, cmap="gray")
plt.title("MRI")
plt.axis("off")


plt.subplot(1,3,2)
plt.imshow(label_slice, cmap="gray")
plt.title("Left Atrium Mask")
plt.axis("off")


plt.subplot(1,3,3)
plt.imshow(image_slice, cmap="gray")
plt.imshow(label_slice, cmap="jet", alpha=0.4)
plt.title("MRI + Mask")
plt.axis("off")


plt.show()

print(metadata.keys())

print(metadata["affine"])
print(metadata["space"])