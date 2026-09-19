from pathlib import Path

import yaml
import torch
import nibabel as nib
import numpy as np

from monai.data import Dataset, DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped
)

from monai.inferers import sliding_window_inference

from model import get_model



# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "config.yaml"
)


IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "Task02_Heart"
    / "imagesTr"
    / "la_024.nii.gz"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_PATH = (
    OUTPUT_DIR
    / "la_024_prediction.nii.gz"
)



# ============================================================
# Load config
# ============================================================

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)



DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Using device:", DEVICE)



# ============================================================
# Preprocessing
# ============================================================

transforms = Compose(
    [

        LoadImaged(
            keys=["image"]
        ),


        EnsureChannelFirstd(
            keys=["image"]
        ),


        EnsureTyped(
            keys=["image"],
            track_meta=True
        ),


        Orientationd(
            keys=["image"],
            axcodes="RAS"
        ),


        Spacingd(
            keys=["image"],
            pixdim=(
                1.25,
                1.25,
                1.25
            ),
            mode="bilinear"
        ),


        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True
        )

    ]
)



# ============================================================
# Dataset
# ============================================================

dataset = Dataset(

    data=[
        {
            "image": str(IMAGE_PATH)
        }
    ],

    transform=transforms

)


loader = DataLoader(
    dataset,
    batch_size=1
)



# ============================================================
# Load model
# ============================================================

model = get_model(
    config
)


checkpoint = torch.load(

    PROJECT_ROOT
    /
    "checkpoints"
    /
    "best_model.pth",

    map_location=DEVICE

)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.to(
    DEVICE
)


model.eval()


print("Model loaded")



# ============================================================
# Inference
# ============================================================

with torch.no_grad():


    for batch in loader:


        image = batch["image"].to(
            DEVICE
        )


        output = sliding_window_inference(

            inputs=image,

            roi_size=tuple(
                config["patch"]["size"]
            ),

            sw_batch_size=1,

            predictor=model

        )


        prediction = torch.argmax(

            output,

            dim=1

        )


        prediction = (
            prediction
            .cpu()
            .numpy()
            .astype(np.uint8)
        )


        # remove batch dimension

        prediction = prediction[0]



        # ====================================================
        # Save NIfTI
        # ====================================================

        original = nib.load(
            str(IMAGE_PATH)
        )


        prediction_img = nib.Nifti1Image(

            prediction,

            original.affine

        )


        nib.save(

            prediction_img,

            str(OUTPUT_PATH)

        )


        print(
            "Prediction saved:"
        )

        print(
            OUTPUT_PATH
        )



print("Inference finished")