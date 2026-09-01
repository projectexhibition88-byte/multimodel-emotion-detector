import os
import sys
from multiprocessing import freeze_support

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, models, transforms


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.0001
NUM_WORKERS = 0  # IMPORTANT: Keep 0 on Windows to avoid multiprocessing errors
NUM_CLASSES = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA PATHS
# ============================================================

FER_TRAIN = "datasets/fer2013/train"
FER_TEST = "datasets/fer2013/test"

RAF_TRAIN = "datasets/raf_db/train"
RAF_TEST = "datasets/raf_db/test"


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CREATE DATASETS
# ============================================================

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

RAF_DB_INDEX_MAP = {
    '1': EMOTIONS.index('surprise'),  # 6
    '2': EMOTIONS.index('fear'),      # 2
    '3': EMOTIONS.index('disgust'),   # 1
    '4': EMOTIONS.index('happy'),     # 3
    '5': EMOTIONS.index('sad'),       # 5
    '6': EMOTIONS.index('angry'),     # 0
    '7': EMOTIONS.index('neutral')    # 4
}


# ============================================================
# CREATE DATASETS
# ============================================================

def load_dataset(path, transform, dataset_name, is_raf_db=False):

    if not os.path.exists(path):
        print(f"\nWARNING: {dataset_name} not found!")
        print(f"Expected path: {path}")
        return None

    try:
        dataset = datasets.ImageFolder(
            root=path,
            transform=transform
        )

        if is_raf_db:
            remapped_samples = []
            for file_path, class_idx in dataset.samples:
                folder_name = dataset.classes[class_idx]
                correct_idx = RAF_DB_INDEX_MAP.get(folder_name, class_idx)
                remapped_samples.append((file_path, correct_idx))

            dataset.samples = remapped_samples
            dataset.targets = [s[1] for s in remapped_samples]
            dataset.classes = EMOTIONS
            print(f"{dataset_name}: {len(dataset)} images (Remapped to standard 7 classes)")
        else:
            print(f"{dataset_name}: {len(dataset)} images")
            print(f"Classes: {dataset.classes}")

        return dataset

    except Exception as error:
        print(f"\nCould not load {dataset_name}")
        print(error)
        return None



# ============================================================
# MODEL
# ============================================================

def create_model(num_classes):

    print("\nLoading ResNet18 model...")

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    # Replace final layer for emotion classification
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )

    return model


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total_predictions += labels.size(0)

        correct_predictions += (
            predicted == labels
        ).sum().item()

    epoch_loss = (
        running_loss / len(train_loader.dataset)
    )

    epoch_accuracy = (
        100 * correct_predictions / total_predictions
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATE MODEL
# ============================================================

def validate(
    model,
    test_loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total_predictions += labels.size(0)

            correct_predictions += (
                predicted == labels
            ).sum().item()

    validation_loss = (
        running_loss / len(test_loader.dataset)
    )

    validation_accuracy = (
        100 * correct_predictions / total_predictions
    )

    return validation_loss, validation_accuracy


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("FACIAL EMOTION MODEL TRAINING")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    print("\nLoading datasets...\n")

    # ----------------------------
    # FER2013
    # ----------------------------

    fer_train = load_dataset(
        FER_TRAIN,
        train_transform,
        "FER2013 Training"
    )

    fer_test = load_dataset(
        FER_TEST,
        test_transform,
        "FER2013 Testing"
    )

    # ----------------------------
    # RAF-DB
    # ----------------------------

    raf_train = load_dataset(
        RAF_TRAIN,
        train_transform,
        "RAF-DB Training",
        is_raf_db=True
    )

    raf_test = load_dataset(
        RAF_TEST,
        test_transform,
        "RAF-DB Testing",
        is_raf_db=True
    )

    # Check that datasets loaded successfully
    if fer_train is None and raf_train is None:
        print("\nERROR: No training datasets could be loaded.")
        return

    # ========================================================
    # COMBINE TRAINING DATASETS
    # ========================================================

    train_datasets = []

    if fer_train is not None:
        train_datasets.append(fer_train)

    if raf_train is not None:
        train_datasets.append(raf_train)

    combined_train_dataset = ConcatDataset(
        train_datasets
    )

    print("\n" + "=" * 60)
    print(
        f"TOTAL TRAINING IMAGES: "
        f"{len(combined_train_dataset)}"
    )
    print("=" * 60)

    # ========================================================
    # COMBINE TEST DATASETS
    # ========================================================

    test_datasets = []

    if fer_test is not None:
        test_datasets.append(fer_test)

    if raf_test is not None:
        test_datasets.append(raf_test)

    if len(test_datasets) > 0:

        combined_test_dataset = ConcatDataset(
            test_datasets
        )

    else:
        print("\nWARNING: No testing dataset found.")
        return

    print(
        f"TOTAL TEST IMAGES: "
        f"{len(combined_test_dataset)}"
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    print("\nCreating DataLoaders...")

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        combined_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ========================================================
    # DETERMINE NUMBER OF CLASSES
    # ========================================================

    # FER2013 normally has 7 emotion classes
    NUM_CLASSES = 7

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = create_model(
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    best_accuracy = 0.0

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print("-" * 60)

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        validation_loss, validation_accuracy = validate(
            model,
            test_loader,
            criterion
        )

        scheduler.step(
            validation_accuracy
        )

        # Save history
        history["epoch"].append(
            epoch + 1
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["validation_accuracy"].append(
            validation_accuracy
        )

        print(
            f"Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%",
            flush=True
        )
        print(
            f"Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.2f}%",
            flush=True
        )

        # Save best model
        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            os.makedirs(
                "models",
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                "models/facial_emotion_model_cuda.pth"
            )

            print(
                f" [+] [★] Best model saved! (Validation Accuracy: {best_accuracy:.2f}%)",
                flush=True
            )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    history_dataframe = pd.DataFrame(
        history
    )

    history_dataframe.to_csv(
        "models/training_history.csv",
        index=False
    )

    # Save final model
    torch.save(
        model.state_dict(),
        "models/facial_emotion_model_final.pth"
    )

    print("\n" + "=" * 60, flush=True)
    print("TRAINING COMPLETED!", flush=True)
    print("=" * 60, flush=True)

    print(
        f"\nBest Validation Accuracy: "
        f"{best_accuracy:.2f}%\n",
        flush=True
    )

    print(
        "\nModels saved inside the "
        "'models' folder."
    )


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()