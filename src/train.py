from pathlib import Path

import torch
import yaml

from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from dataset import get_dataloaders
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


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)



# ============================================================
# Configuration
# ============================================================

set_determinism(
    seed=42
)


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)



DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Using device:",
    DEVICE
)



# ============================================================
# Training parameters
# ============================================================

NUM_CLASSES = 2


PATCH_SIZE = tuple(
    config["patch"]["size"]
)


EPOCHS = config["training"]["epochs"]


LEARNING_RATE = config["training"]["learning_rate"]


BATCH_SIZE = config["training"]["batch_size"]



print(
    "Loading dataloaders..."
)


train_loader, val_loader = get_dataloaders(
    config
)



print(
    "Training cases:",
    len(train_loader.dataset)
)


print(
    "Validation cases:",
    len(val_loader.dataset)
)



# ============================================================
# Model
# ============================================================

model = get_model(
    config
)


model = model.to(
    DEVICE
)


print(model)



# ============================================================
# Loss
# ============================================================

loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True
)



# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)



# ============================================================
# AMP
# ============================================================

scaler = torch.cuda.amp.GradScaler()



# ============================================================
# Metric
# ============================================================

dice_metric = DiceMetric(
    include_background=False,
    reduction="mean"
)



# ============================================================
# Validation
# ============================================================

def validate():

    model.eval()

    dice_metric.reset()


    with torch.no_grad():

        for batch in tqdm(
            val_loader,
            desc="Validation"
        ):


            images = batch["image"].to(
                DEVICE
            )

            labels = batch["label"].to(
                DEVICE
            )


            outputs = sliding_window_inference(

                inputs=images,

                roi_size=PATCH_SIZE,

                sw_batch_size=1,

                predictor=model

            )


            predictions = torch.argmax(
                outputs,
                dim=1,
                keepdim=True
            )


            dice_metric(
                y_pred=predictions,
                y=labels
            )



    dice = dice_metric.aggregate().item()


    return dice



# ============================================================
# Training loop
# ============================================================

best_dice = 0.0



for epoch in range(EPOCHS):


    print("\n")
    print("=" * 60)

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )

    print("=" * 60)



    model.train()


    epoch_loss = 0



    progress = tqdm(
        train_loader,
        desc="Training"
    )



    for batch in progress:


        images = batch["image"].to(
            DEVICE
        )

        labels = batch["label"].to(
            DEVICE
        )



        optimizer.zero_grad()



        with torch.cuda.amp.autocast():

            outputs = model(
                images
            )


            loss = loss_function(
                outputs,
                labels
            )



        scaler.scale(
            loss
        ).backward()



        scaler.step(
            optimizer
        )


        scaler.update()



        epoch_loss += loss.item()



        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )



    epoch_loss /= len(train_loader)



    print(
        f"Training loss: {epoch_loss:.4f}"
    )



    # -------------------------
    # Validation
    # -------------------------

    dice = validate()



    print(
        f"Validation Dice: {dice:.4f}"
    )



    # -------------------------
    # Save best
    # -------------------------

    if dice > best_dice:


        best_dice = dice


        torch.save(

            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "dice":
                    best_dice
            },

            CHECKPOINT_PATH

        )


        print(
            "Saved new best model!"
        )



print("\nTraining finished.")

print(
    "Best Dice:",
    best_dice
)