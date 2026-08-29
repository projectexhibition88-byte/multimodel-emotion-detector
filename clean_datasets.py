from pathlib import Path
import cv2
import hashlib
import shutil


# ============================================================
# Supported image formats
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# File Hash
# ============================================================

def get_file_hash(file_path):
    """Create a hash for detecting exact duplicate files."""

    hasher = hashlib.md5()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================================
# Move file while preserving folder structure
# ============================================================

def move_preserving_structure(
    source,
    dataset_root,
    destination_root
):
    """
    Moves a file while preserving its original folder structure.

    This prevents files with the same name from overwriting
    each other.
    """

    relative_path = source.relative_to(
        dataset_root
    )

    destination = destination_root / relative_path

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.move(
        str(source),
        str(destination)
    )


# ============================================================
# Clean normal datasets
# FER2013 and RAF-DB
# ============================================================

def clean_normal_dataset(dataset_path):
    """
    Cleans normal image datasets such as FER2013 and RAF-DB.

    It checks recursively inside train/test and all class folders.
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():

        print(
            f"\n❌ DATASET NOT FOUND: {dataset_path}"
        )

        return

    duplicates_folder = (
        dataset_path / "_cleaning_duplicates"
    )

    corrupted_folder = (
        dataset_path / "_cleaning_corrupted"
    )

    seen_hashes = set()

    total_images = 0
    corrupted_images = 0
    too_small_images = 0
    duplicate_images = 0

    print(
        "\n" + "=" * 65
    )

    print(
        f"CLEANING: {dataset_path}"
    )

    print(
        "=" * 65
    )

    for image_path in dataset_path.rglob("*"):

        # Skip folders
        if not image_path.is_file():
            continue

        # Skip non-image files
        if (
            image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        # Skip files already moved during previous cleaning
        if (
            "_cleaning_duplicates"
            in image_path.parts
        ):
            continue

        if (
            "_cleaning_corrupted"
            in image_path.parts
        ):
            continue

        total_images += 1

        try:

            # ------------------------------------------------
            # Read image
            # ------------------------------------------------

            image = cv2.imread(
                str(image_path)
            )

            # ------------------------------------------------
            # Check corrupted images
            # ------------------------------------------------

            if image is None:

                move_preserving_structure(
                    image_path,
                    dataset_path,
                    corrupted_folder
                )

                corrupted_images += 1

                print(
                    f"CORRUPTED: {image_path}"
                )

                continue

            height, width = image.shape[:2]

            # ------------------------------------------------
            # Check extremely small images
            # ------------------------------------------------

            if (
                height < 20
                or width < 20
            ):

                move_preserving_structure(
                    image_path,
                    dataset_path,
                    corrupted_folder
                )

                too_small_images += 1

                print(
                    f"TOO SMALL: {image_path}"
                )

                continue

            # ------------------------------------------------
            # Check exact duplicates
            # ------------------------------------------------

            image_hash = get_file_hash(
                image_path
            )

            if image_hash in seen_hashes:

                move_preserving_structure(
                    image_path,
                    dataset_path,
                    duplicates_folder
                )

                duplicate_images += 1

                print(
                    f"DUPLICATE: {image_path}"
                )

            else:

                seen_hashes.add(
                    image_hash
                )

        except Exception as error:

            print(
                f"ERROR CHECKING: {image_path}"
            )

            print(
                f"Reason: {error}"
            )

    # ========================================================
    # Results
    # ========================================================

    print(
        "\nCLEANING RESULTS"
    )

    print(
        "-" * 65
    )

    print(
        f"Total images checked: "
        f"{total_images}"
    )

    print(
        f"Corrupted images moved: "
        f"{corrupted_images}"
    )

    print(
        f"Too-small images moved: "
        f"{too_small_images}"
    )

    print(
        f"Exact duplicates moved: "
        f"{duplicate_images}"
    )


# ============================================================
# RUN FER2013 + RAF-DB CLEANING
# ============================================================

if __name__ == "__main__":

    print(
        "\nSTARTING DATASET CLEANING...\n"
    )

    # --------------------------------------------------------
    # 1. FER2013
    # --------------------------------------------------------

    clean_normal_dataset(
        "datasets/fer2013"
    )

    # --------------------------------------------------------
    # 2. RAF-DB
    # --------------------------------------------------------

    clean_normal_dataset(
        "datasets/raf_db"
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "FER2013 + RAF-DB CLEANING COMPLETED!"
    )

    print(
        "=" * 65
    )