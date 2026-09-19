from IPython.core.magics import config
from pathlib import Path
import random


from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
)


from monai.data import (
    CacheDataset,
    Dataset,
    DataLoader,
    list_data_collate,
)



def get_dataloaders(config):
    """
    Creates training and validation DataLoaders.

    Returns:
        train_loader
        val_loader
    """


    # ==========================================
    # Dataset paths
    # ==========================================

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    data_dir = PROJECT_ROOT / config["data"]["root"]


    images_dir = data_dir / "imagesTr"
    labels_dir = data_dir / "labelsTr"



    image_files = sorted(
        images_dir.glob("*.nii.gz")
    )

    label_files = sorted(
        labels_dir.glob("*.nii.gz")
    )


    print(f"Found {len(image_files)} images")
    print(f"Found {len(label_files)} labels")



    # ==========================================
    # Create MONAI dictionaries
    # ==========================================

    data_dicts = []


    for image, label in zip(
        image_files,
        label_files
    ):

        data_dicts.append(
            {
                "image": str(image),
                "label": str(label)
            }
        )



    # ==========================================
    # Train / validation split
    # ==========================================

    random.seed(
        config["system"]["seed"]
    )


    random.shuffle(data_dicts)



    train_files = data_dicts[:16]

    val_files = data_dicts[16:]



    print(
        f"Training cases: {len(train_files)}"
    )

    print(
        f"Validation cases: {len(val_files)}"
    )



    # ==========================================
    # Deterministic transforms
    # ==========================================

    base_transforms = Compose(
        [

            LoadImaged(
                keys=[
                    "image",
                    "label"
                ]
            ),


            EnsureChannelFirstd(
                keys=[
                    "image",
                    "label"
                ]
            ),


            Orientationd(
                keys=[
                    "image",
                    "label"
                ],
                axcodes="RAS",
                labels=None
            ),


            Spacingd(
                keys=[
                    "image",
                    "label"
                ],
                pixdim=(
                    1.25,
                    1.25,
                    1.25
                ),
                mode=(
                    "bilinear",
                    "nearest"
                )
            ),


            NormalizeIntensityd(
                keys=[
                    "image"
                ],
                nonzero=True,
                channel_wise=True
            ),

        ]
    )



    # ==========================================
    # Random training transforms
    # ==========================================

    train_random_transforms = Compose(
        [

            RandCropByPosNegLabeld(
                keys=[
                    "image",
                    "label"
                ],

                label_key="label",

                spatial_size=tuple(
                    config["patch"]["size"]
                ),

                pos=1,
                neg=1,

                num_samples=config["patch"]["samples_per_image"]
            )

        ]
    )



    # ==========================================
    # Cache deterministic operations
    # ==========================================

    train_cache = CacheDataset(
        data=train_files,
        transform=base_transforms,
        cache_rate=1.0,
        num_workers=config["system"]["num_workers"]
    )



    val_ds = CacheDataset(
        data=val_files,
        transform=base_transforms,
        cache_rate=1.0,
        num_workers=config["system"]["num_workers"]
    )



    # ==========================================
    # Add random crops after cache
    # ==========================================

    train_ds = Dataset(
        data=train_cache,
        transform=train_random_transforms
    )



    # ==========================================
    # DataLoaders
    # ==========================================

    train_loader = DataLoader(
        train_ds,

        batch_size=config["training"]["batch_size"],

        shuffle=True,

        num_workers=0,

        collate_fn=list_data_collate
    )



    val_loader = DataLoader(
        val_ds,

        batch_size=1,

        shuffle=False,

        num_workers=0
    )



    return train_loader, val_loader