import torch
import torch.nn as nn
import torchvision
from transformers import CLIPModel, CLIPProcessor



class SimpleCNN(nn.Module):
    def __init__(self):
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
        self.fc1 = nn.Linear(256 * 32 * 32, 1000)  
        self.fc2 = nn.Linear(1000, 1)

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
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, embed):
        return self.layers(embed)

def jpegcompression_loss_fn(target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None,
                     reward_model_resume_from=None):
    scorer = JpegCompressionScorer(dtype=torch_dtype, model_path=reward_model_resume_from).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    def loss_fn(im_pix_un): 
        if accelerator.mixed_precision == "fp16":
            with accelerator.autocast():
                rewards = scorer(im_pix_un)
        else:
            rewards = scorer(im_pix_un)
        
        if target is None:
            loss = rewards
        else:
            loss = abs(rewards - target)
        return loss * grad_scale, rewards
    return loss_fn

class JpegCompressionScorer(nn.Module):
    def __init__(self, dtype=None, model_path = None):
        super().__init__()
        # self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # self.clip.requires_grad_(False)
        # self.score_generator = MLP()
        self.score_generator = SimpleCNN()
        if model_path:
            state_dict = torch.load(model_path)
            self.score_generator.load_state_dict(state_dict)
        if dtype:
            self.dtype = dtype
        self.target_size = 512
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
       

    def set_device(self, device, inference_type):
        # self.clip.to(device, dtype = inference_type)
        self.score_generator.to(device) # , dtype = inference_type

    def __call__(self, images):
        device = next(self.parameters()).device
        im_pix = torchvision.transforms.Resize(self.target_size)(images)
        im_pix = self.normalize(im_pix).to(images.dtype)
        # embed = self.clip.get_image_features(pixel_values=im_pix)
        # embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.score_generator(im_pix).squeeze(1)

        


    

