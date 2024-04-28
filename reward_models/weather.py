import torch
import torch.nn as nn
import torchvision
from transformers import CLIPModel, CLIPProcessor

class SimpleCNN(nn.Module): # parameter = 6333513
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(64 * 14 * 14, 500)  
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = self.layer1(x)
        # print("x1", x.shape)
        x = self.layer2(x)
        # print("x2", x.shape)
        x = self.layer3(x)
        # print("x3", x.shape)
        x = self.layer4(x)
        # print("x4", x.shape)

        x = x.reshape(x.size(0), -1)
        # print("x reshape", x.shape)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, embed):
        return self.layers(embed)


def weather_loss_fn(target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None,
                     reward_model_resume_from=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = WeatherScorer(dtype=torch_dtype, model_path=reward_model_resume_from).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    def loss_fn(im_pix_un): 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix_un)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        if accelerator.mixed_precision == "fp16":
            with accelerator.autocast():
                rewards = scorer(im_pix)
        else:
            rewards = scorer(im_pix)
        
        if target is None:
            loss = rewards
        else:
            loss = abs(rewards - target)
        return loss * grad_scale, rewards
    return loss_fn

class WeatherScorer(nn.Module):    # Reward model
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
        self.target_size = 224  # mlp 224, cnn 512 (use 224 for both...?)
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])

    def set_device(self, device, inference_type):
        # self.clip.to(device, dtype = inference_type)    # uncomment for mlp
        self.score_generator.to(device) #  dtype = inference_dtype

    def __call__(self, images):
        device = next(self.parameters()).device
        im_pix = torchvision.transforms.Resize(self.target_size)(images)
        im_pix = self.normalize(im_pix).to(images.dtype)
        # embed = self.clip.get_image_features(pixel_values=im_pix)
        # embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        # return self.score_generator(embed).squeeze(1)   # MLP
        return self.score_generator(im_pix).squeeze(1)    # for simpleCNN
    


# class ClipEmbed(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

#     def __call__(self, images):
#         with torch.no_grad():
#             param = next(self.parameters()) 
#             device = param.device
#             dtype = param.dtype
#             inputs = self.processor(images=images, return_tensors="pt")
#             inputs = {k: v.to(dtype).to(device) for k, v in inputs.items()}
#             embed = self.clip.get_image_features(**inputs)
#             embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
#             return embed
        
# class Weather():    # Reward model
#     def __init__(self):
#         self.feature_extractor = ClipEmbed()
#         self.score_generator = MLP()

#     def set_device(self, device, inference_dtype=None):
#         self.feature_extractor.to(device, dtype = inference_dtype)
#         self.score_generator.to(device) #  dtype = inference_dtype
    
#     def generate_score(self, image):
#         out_embed = self.feature_extractor(image)
#         score = self.score_generator(out_embed)
#         return score

#     def load_weight(self, model_path):
#         state_dict = torch.load(model_path)
#         self.score_generator.load_state_dict(state_dict)
