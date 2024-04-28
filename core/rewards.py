from PIL import Image
import io
import numpy as np
import numpy.random as random
import torch
import albumentations as A

def piexlate(device):
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        level_ids = []
        transform_images = []
        level_choice = [i for i in range(1,20)]
        for i, image in enumerate(images):
            level_idx = random.choice(range(len(level_choice)))
            level = level_choice[level_idx]
            width, height = image.size
            block_size = max(1, level)
            image_small = image.resize((width // block_size, height // block_size), Image.BILINEAR)
            image_large = image_small.resize(image.size, Image.NEAREST)
            level_ids.append(level_idx)
            transform_images.append(image_large)
        transform_images_tensor = torch.Tensor(np.array(transform_images)).to(device)
        transform_images_tensor = (transform_images_tensor.permute(0,3,1,2) / 255).clamp(0,1)
        return np.array(level_ids), transform_images_tensor, {}

    return _fn


def snow(device):
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        # images = [Image.fromarray(image) for image in images]
        # level = [torch.random() for _ in images]
        levels = []
        transform_images = []
        # rain_type_dict = {0: None, 1: "drizzle", 2: "heavy", 3:"torrential"}
        for i, image in enumerate(images):
            level = torch.rand(1).item()
            # print(level)
            # import pdb; pdb.set_trace()
            # level = torch.randint(0, 100, (1,)).item()
            # level = torch.randint(0, 4, (1,)).item()
            # if level == 0:
            #     transform = A.Compose([
            #         A.RandomBrightnessContrast(p=0.5),
            #         A.PixelDropout(p=0.5),
            #     ])

            # else :
            #     transform = A.Compose([
            #             # A.RandomSnow(brightness_coeff=2.5, snow_point_lower=level, snow_point_upper=level, p=1)
            #             A.RandomRain(drop_length = level, rain_type = rain_type_dict[level], p=1)
            #     ])
            transform = A.Compose([
                        A.RandomSnow(brightness_coeff=2.5, snow_point_lower=level, snow_point_upper=level, p=1)
            ])
            levels.append(level)
            transform_images.append(transform(image=image)['image'])
        # print("levels", levels)
        # transform_images = [transform(image=image)['image'] for image in images]
        transform_images_tensor = torch.Tensor(np.array(transform_images)).to(device)
        transform_images_tensor = (transform_images_tensor.permute(0,3,1,2) / 255).clamp(0,1)
        # how much snow was added for each image
        # snow_points = [metadata for _ in range(len(images))]
        return np.array(levels), transform_images_tensor, {}

    return _fn


def jpeg_incompressibility(device):
    def _fn(images, prompts, metadata):
        org_type = images.dtype
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        # images = [Image.fromarray(image) for image in images]
        
        transform = A.Compose([
                A.Blur(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.PixelDropout(p=0.5),
                A.ImageCompression(quality_lower=10, quality_upper=80, p=1)
        ])
        transform_images = [transform(image=image)['image'] for image in images]
        transform_images_tensor = torch.Tensor(np.array(transform_images)).to(device, dtype=org_type)
        transform_images_tensor = (transform_images_tensor.permute(0,3,1,2) / 255).clamp(0,1)
        transform_images_pil = [Image.fromarray(image) for image in transform_images]
        buffers = [io.BytesIO() for _ in transform_images_pil]
        for image, buffer in zip(transform_images_pil, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), transform_images_tensor, {}

    return _fn


def jpeg_compressibility(device):
    def _fn(images, prompts, metadata):
        org_type = images.dtype
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        transform = A.Compose([
                A.Blur(p=0.5),
                A.Defocus(p=0.2),
                A.ImageCompression(quality_lower=10, quality_upper=100, p=1)
        ])
        transform_images = [transform(image=image)['image'] for image in images]
        transform_images_tensor = torch.Tensor(np.array(transform_images)).to(device, dtype=org_type)
        transform_images_tensor = (transform_images_tensor.permute(0,3,1,2) / 255).clamp(0,1)
        transform_images_pil = [Image.fromarray(image) for image in transform_images]
        buffers = [io.BytesIO() for _ in transform_images_pil]
        for image, buffer in zip(transform_images_pil, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), transform_images_tensor, {}

    return _fn