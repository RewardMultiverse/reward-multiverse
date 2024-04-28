import cv2
from matplotlib import pyplot as plt
import albumentations as A
from pathlib import Path
import json

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

def load_rgb_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == "__main__":
    # unsplash_dataset_path = Path("unsplash-dataset-lite-latest/examples")
    unsplash_dataset_path = Path("/projects/katefgroup/datasets/coco17/train2017/")
    save_root_path = Path("unsplash-dataset-lite-latest/augmented")

    supported_formats = (".jpg", ".jpeg", ".png")

    # Define different transformations
    transformations = [
        ["GaussianBlur_11", [A.GaussianBlur(blur_limit=(11, 11), p=1)]],
        ["GaussianBlur_55", [A.GaussianBlur(blur_limit=(55, 55), p=1)]],
        ["GaussianBlur_155", [A.GaussianBlur(blur_limit=(155, 155), p=1)]]
    ]

    augmented_info = []
    for image_path in unsplash_dataset_path.glob("*"):
        print(image_path)
        if image_path.suffix.lower() not in supported_formats:
            continue

        image = load_rgb_image(str(image_path))
        visualize(image)
        plt.show()
        break

        # Apply and save different transformations
        for transform_name, transform in transformations:
            augmented_image = image.copy()
            for t in transform:
                augmented_image = t(image=augmented_image)["image"]

            # Save augmented image
            save_folder = save_root_path / transform_name
            save_folder.mkdir(parents=True, exist_ok=True)
            save_image_path = save_folder / image_path.name
            cv2.imwrite(str(save_image_path), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            print(f"Saved augmented image to {save_image_path}")

            # Store transformation info
            transformation_info = [{"type": t.__class__.__name__, "parameters": t.get_params()} for t in transform]
            augmented_info.append({
                "original_image_path": str(image_path),
                "augmented_image_path": str(save_image_path),
                "transform": transformation_info,
            })

    # Save augmented info to a JSON file
    # json_path = save_root_path / "augmented_info.json"
    # with open(json_path, "w") as f:
    #     json.dump(augmented_info, f, default=str, indent=4)
    # print(f"Saved augmented info to {json_path}")
