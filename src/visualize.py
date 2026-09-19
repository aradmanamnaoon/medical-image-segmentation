from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import zoom



PROJECT_ROOT = Path(__file__).resolve().parent.parent


IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "Task02_Heart"
    / "imagesTr"
    / "la_024.nii.gz"
)


LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "Task02_Heart"
    / "labelsTr"
    / "la_024.nii.gz"
)


PRED_PATH = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "la_024_prediction.nii.gz"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "la_024_result.png"
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)



# ============================================================
# Load data
# ============================================================

image = nib.load(
    str(IMAGE_PATH)
).get_fdata()


label = nib.load(
    str(LABEL_PATH)
).get_fdata()


prediction = nib.load(
    str(PRED_PATH)
).get_fdata()



prediction = np.squeeze(prediction)


label = (label > 0).astype(np.uint8)

prediction = (prediction > 0).astype(np.uint8)



print("Image:", image.shape)
print("Label:", label.shape)
print("Prediction before resize:", prediction.shape)



# ============================================================
# Resize prediction to label size
# ============================================================

if prediction.shape != label.shape:

    resize_factor = (

        label.shape[0] / prediction.shape[0],

        label.shape[1] / prediction.shape[1],

        label.shape[2] / prediction.shape[2],

    )


    prediction = zoom(
        prediction,
        resize_factor,
        order=0
    )


prediction = (
    prediction > 0
).astype(np.uint8)


print(
    "Prediction after resize:",
    prediction.shape
)



# ============================================================
# Dice
# ============================================================

intersection = np.sum(
    (label == 1)
    &
    (prediction == 1)
)


dice = (
    2.0 * intersection
    /
    (
        np.sum(label)
        +
        np.sum(prediction)
    )
)


print(
    f"Dice: {dice:.4f}"
)



# ============================================================
# Select slice
# ============================================================

slice_idx = np.argmax(
    np.sum(label, axis=(0,1))
)


print(
    "Slice:",
    slice_idx
)



image_slice = image[:,:,slice_idx]

label_slice = label[:,:,slice_idx]

prediction_slice = prediction[:,:,slice_idx]



# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15,5)
)



axes[0].imshow(
    image_slice.T,
    cmap="gray",
    origin="lower"
)

axes[0].set_title(
    "MRI"
)

axes[0].axis("off")



axes[1].imshow(
    image_slice.T,
    cmap="gray",
    origin="lower"
)

axes[1].imshow(
    label_slice.T,
    alpha=0.5,
    origin="lower"
)

axes[1].set_title(
    "Ground Truth"
)

axes[1].axis("off")



axes[2].imshow(
    image_slice.T,
    cmap="gray",
    origin="lower"
)

axes[2].imshow(
    prediction_slice.T,
    alpha=0.5,
    origin="lower"
)

axes[2].set_title(
    "Prediction"
)

axes[2].axis("off")



plt.suptitle(
    f"la_024 | Dice={dice:.4f}"
)


plt.tight_layout()


plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved:",
    OUTPUT_PATH
)