import torch
import torch.nn as nn
import torchvision
from transformers import CLIPModel, CLIPProcessor

class SimpleCNN(nn.Module): # parameter = 6333513
    def __init__(self, num_class):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(256 * 32 * 32, 512)  
        self.fc2 = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1000, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.Linear(16, num_class),
        )

    def forward(self, embed):
        return self.layers(embed)

def piexlate_loss_fn(target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None,
                     reward_model_resume_from=None,
                     num_of_labels=None):
    scorer = PixelateScorer(dtype=torch_dtype, model_path=reward_model_resume_from, num_class=num_of_labels).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    def loss_fn(im_pix_un): 
        if accelerator.mixed_precision == "fp16":
            with accelerator.autocast():
                rewards = scorer(im_pix_un)
        else:
            rewards = scorer(im_pix_un)

        target_tensors = torch.full((rewards.shape[0],), target-1).to(rewards.device, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(rewards, target_tensors)
        return loss * grad_scale, nn.Softmax(dim=-1)(rewards)
    return loss_fn

class PixelateModel(nn.Module):
    def __init__(self, num_class = None):
        super().__init__()
        self.embed_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.score_model = MLP(num_class)
    def __call__(self, im):
        return self.score_model(self.embed_model(im))


class PixelateScorer(nn.Module):    # Reward model
    def __init__(self, dtype=None, model_path = None, num_class = None):
        super().__init__()
        # self.score_generator = SimpleCNN(num_class)
        self.score_generator = PixelateModel(num_class)
        if model_path:
            state_dict = torch.load(model_path)
            self.score_generator.load_state_dict(state_dict)
            self.score_generator.requires_grad_(False)
        else:
            self.score_generator.requires_grad_(True)
        if dtype:
            self.dtype = dtype
        self.target_size = 224  # mlp 224, cnn 512 (use 224 for both...?)
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])

    def set_device(self, device, inference_type):
        # self.embed_model.to(device, dtype=inference_type)
        # self.clip.to(device, dtype = inference_type)    # uncomment for mlp
        self.score_generator.to(device) #  dtype = inference_dtype

    def __call__(self, images):
        device = next(self.parameters()).device
        im_pix = torchvision.transforms.Resize(self.target_size)(images)
        im_pix = self.normalize(im_pix).to(images.dtype)
        # import pdb
        # pdb.set_trace()
        # embed = self.embed_model(im_pix)
        # embed = self.clip.get_image_features(pixel_values=im_pix)
        # embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        # return self.score_generator(embed).squeeze(1)   # MLP
        return self.score_generator(im_pix).squeeze(1)    # for simpleCNN
