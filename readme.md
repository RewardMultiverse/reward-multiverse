# Reward Ｍultiverse (RM)
Implementing Various Reward Functions to Align Diffusion Models with Different Applications by Cheng An Hsieh, Benjamin Chiang, Jennifer Wang, Mihir Prabhudesai.
<div align="center">

<!-- TITLE -->
![Reward Ｍultiverse](https://rewardmultiverse.github.io/images/method.jpg)

[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](https://rewardmultiverse.github.io/)
</div>


# Abstract
In recent advancements in text-to-image synthesis, fine-tuning diffusion models through reward-driven backpropagation has shown promising results. In this work, we introduce a framework for implementing and applying various reward functions to align text-to-image diffusion models according to specific visual characteristics. Our methodology involves training reward models capable of distinguishing unique image features—such as the presence of snow, rain, and pixelate—and using these models to guide the diffusion process towards generating images with desired attributes. We present a versatile tool that allows users to create custom reward models, facilitating personalized image generation. Through extensive experiments, we demonstrate the effectiveness of our reward models in producing high-fidelity, attribute-specific images. Our work not only extends the capabilities of text-to-image models but also provides a scalable platform for community-driven enhancements in image generation.

## Reward Functions:

| Snow | Rain | 
| -------- | -------- | 
| ![Snow](https://github.com/RewardMultiverse/reward-multiverse/blob/main/images/snow.gif)     | ![Rain](https://github.com/RewardMultiverse/reward-multiverse/blob/main/images/rain.gif)     |

| Pixelate | Image Compression | 
| -------- | -------- | 
| ![Pixelate](https://github.com/RewardMultiverse/reward-multiverse/blob/main/images/wolf.gif)     | ![Compression](https://github.com/RewardMultiverse/reward-multiverse/blob/main/images/bird.gif)     |

| Day&Night |
| -------- | 
| ![DayNight](https://github.com/RewardMultiverse/reward-multiverse/blob/main/images/daynight.gif)     |

## Model Weight: 
Access the model weights [here](https://drive.google.com/drive/folders/1V4Pr55-Jkxqgqa0rmYCLt8zXRCSqsR9M?usp=sharing).

## Code

### Installation 
Create a conda environment with the following command:
```bash
conda create -n reward python=3.10
conda activate reward
pip install -r requirements.txt
```

## Citation
If you find our work useful, please consider citing:
```bibtex
@software{rewardmultiverse2024,
  title={Implementing Various Reward Functions to Align Diffusion Models with Different Applications},
  author={Hsieh, Cheng An and Chiang, Benjamin and Wang, Jennifer and Prabhudesai, Mihir},
  year={2024},
  url={https://github.com/rewardUniverse/reward-multiverse},
  note={Accessed: 2024}
}



