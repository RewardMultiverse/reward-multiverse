import torch
import os
from ml_collections import config_flags
from reward_models import load_model
from PIL import Image
from absl import app, flags
from torchvision import transforms

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/test_reward.py", "Training configuration.")
def main(_):
    config = FLAGS.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reward_model = load_model(config, None, None)
    state_dict = torch.load(config.reward_model_resume_from)
    reward_model.score_generator.load_state_dict(state_dict)
    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    reward_model.set_device(device, inference_dtype)
    reward_model.requires_grad_(False)
    reward_model.score_generator.requires_grad_(False)
    reward_model.eval()
    transform = transforms.ToTensor()

    # Loop through each file in the directory
    for filename in os.listdir(config.image_directories):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            file_path = os.path.join(config.image_directories, filename)
            try:
                img = Image.open(file_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                img_tensor = img_tensor.to(device, dtype=inference_dtype)
                if config.mixed_precision == "fp16":
                    with torch.autocast(device_type="cuda"):
                        score = reward_model(img_tensor)
                else:
                    score = reward_model(img_tensor)
                    score = score.cpu().numpy()
                print(f"File name:{filename}, Score:{score}")
            except IOError:
                print(f"Failed to open {filename}")
    
if __name__ == "__main__":
    app.run(main)