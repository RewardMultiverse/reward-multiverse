import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize
from PIL import Image
import torchvision

"""
Custom dataloader for training datasets with a specific directory structure.

Directory structure should be organized as follows:
- Each class should have its own subdirectory.
- Inside each class subdirectory, it should store all images.

Example structure:
data_dir_name
    |-class1
    |---img1.jpg
    |---img2.jpg
    |---....
    |-class2
    |---img1.jpg
    |---img2.jpg
    |---....
"""

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)

        self.image_paths = []
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            if os.path.isdir(cls_dir):
                self.image_paths.extend([os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if img.endswith(('.png', '.jpg'))])
        # print("self.image_paths", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract label from directory structure
        label = self.extract_label_from_directory(img_path)

        return image, label

    def extract_label_from_directory(self, img_path):
        # Extract label from the directory structure
        label = img_path.split(os.sep)[-2]  # Assuming label is the parent directory name
        return label

def create_custom_dataloader(data_dir, batch_size):
    transform = torchvision.transforms.Compose([
        Resize((32, 32)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
    ])

    dataset = CustomDataset(data_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
