import ml_collections
import os
def general():
    config = ml_collections.ConfigDict()
    # rewards
    config.reward_fn = "jpeg_compressibility"
    config.model_code = "image_compression_reward"
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision  = "fp16"
    config.reward_model_resume_from = "logs/2024.04.04_22.55.43/checkpoints/checkpoint_2/mlp.pt"
    config.image_directories = "data_test_reward/image_compression"
    return config



def get_config(name):
    return globals()[name]()