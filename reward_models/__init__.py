from reward_models.image_compression import *
from reward_models.weather import *
from reward_models.image_pixelate import *

def load_model(config, accelerator=None, dtype=None):
    if "image_compression_reward" in config.model_code:
        return JpegCompressionScorer(dtype)
    elif "image_compression_diff" in config.model_code:
        return jpegcompression_loss_fn(config.target_val, config.grad_scale, accelerator.device, accelerator, dtype, config.reward_model_resume_from)
    elif "weather_diff" in config.model_code:
        print("config", config.model_code)
        return weather_loss_fn(config.target_val, config.grad_scale, accelerator.device, accelerator, dtype, config.reward_model_resume_from)
    elif "weather_reward" in config.model_code:
        return WeatherScorer(dtype)
    elif "image_pixelate_reward" in config.model_code:
        return PixelateScorer(dtype = dtype, num_class = config.num_of_labels)
    elif "image_pixelate_diff" in config.model_code:
        return piexlate_loss_fn(config.target_val, config.grad_scale, accelerator.device, accelerator, dtype, config.reward_model_resume_from, config.num_of_labels)
    else:
        raise NotImplementedError(f"Can't identify reward model:{config.model_code}")