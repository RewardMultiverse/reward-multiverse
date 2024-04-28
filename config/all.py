import ml_collections
# import ipdb 
import os
# st = ipdb.set_trace

def general():
    config = ml_collections.ConfigDict()

   ###### General ######    
    config.eval_prompt_fn = ''
    config.soup_inference = False
    config.all_step_rendering = False
    config.ddpm = False
    config.complex_dist = False
    config.skipt = False
    config.save_freq = 4
    config.resume_from = ""
    config.resume_from_2 = ""
    config.vis_freq = 1
    config.max_vis_images = 4
    config.only_eval = False
    config.decay_lr = False
    config.remove_word = ""
    config.run_name = ""
    config.task = "regression"
    config.loss_fn = "mse" # mse / crossentropy
    config.num_of_labels = 0
    
    # prompting
    config.prompt_fn = "imagenet_animals"
    config.fixed_noise = False
    config.prompt_fn_kwargs = {}

    # rewards
    # reward function to use. see `rewards.py` for available reward functions.
    config.reward_fn = "jpeg_compressibility"
    config.model_code = "image_compression"
    config.debug = False
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision  = "fp16"
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 5
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # random seed for reproducibility.
    config.seed = 42    
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    config.num_epochs = 100    
    # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
    # attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
    # about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
    # files will also be large.
    config.use_lora = True
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True

    config.visualize_train = False
    config.visualize_eval = True

    config.truncated_backprop = False
    config.truncated_backprop_rand = False
    config.truncated_backprop_minmax = (35,45)
    config.trunc_backprop_timestep = 100
    
    config.grad_checkpoint = True
    config.same_evaluation = True
    
    config.min_timesteps = 39
    config.max_timesteps = 49
    
    ###### Training ######    
    config.train = train = ml_collections.ConfigDict()
    train.loss_coeff = 1.0
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8 
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0    

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    # revision of the model to load.
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for sampling.
    sample.batch_size = 1    
    return config



def set_config_batch(config,total_samples_per_epoch, total_batch_size, per_gpu_capacity=1):    
    #  Samples per epoch
    config.train.total_samples_per_epoch = total_samples_per_epoch  #(~~~~ this is desired ~~~~)
    config.train.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    # st()
    assert config.train.total_samples_per_epoch%config.train.num_gpus==0, "total_samples_per_epoch must be divisible by num_gpus"
    config.train.samples_per_epoch_per_gpu = config.train.total_samples_per_epoch//config.train.num_gpus
    
    #  Total batch size
    config.train.total_batch_size = total_batch_size  #(~~~~ this is desired ~~~~)
    # given we have 4 gpus
    assert config.train.total_batch_size%config.train.num_gpus==0, "total_batch_size must be divisible by num_gpus"
    config.train.batch_size_per_gpu = config.train.total_batch_size//config.train.num_gpus
    config.train.batch_size_per_gpu_available = per_gpu_capacity    #(this quantity depends on the gpu used)
    assert config.train.batch_size_per_gpu%config.train.batch_size_per_gpu_available==0, "batch_size_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.gradient_accumulation_steps = config.train.batch_size_per_gpu//config.train.batch_size_per_gpu_available
    
    #  How many data loader iterations to go through each epoch
    assert config.train.samples_per_epoch_per_gpu%config.train.batch_size_per_gpu_available==0, "samples_per_epoch_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.data_loader_iterations  = config.train.samples_per_epoch_per_gpu//config.train.batch_size_per_gpu_available    
    return config

def img_pixelate_reward():
    config = general()
    config.wandb_name = "train_reward_for_img_pixelate"
    config.task = "classification"
    config.loss_fn = "crossentropy"
    config.num_of_labels = 20
    config.debug = False
    config.num_epochs = 200
    config.steps = 50
    config.prompt_fn = "simple_animals"
    config.reward_fn = "piexlate"
    config.eval_prompt_fn = "simple_animals"
    config.model_code = "image_pixelate_reward"
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 1
    config.train.learning_rate = 1e-3
    config.train.adam_weight_decay = 0
    config.save_freq = 20
    config.num_checkpoint_limit = 5    
    config.sd_guidance_scale = 7.5
    config.max_vis_images = 2
    config = set_config_batch(config,total_samples_per_epoch=256,total_batch_size= 64, per_gpu_capacity=8)
    return config


def img_pixelate_diff():
    config = general()
    config.wandb_name = "train_diff_for_img_pixelate"
    config.task = "classification"
    config.loss_fn = "crossentropy"
    config.target_val = 18
    config.num_of_labels = 20
    config.debug = False
    config.num_epochs = 20
    config.steps = 50
    config.grad_scale = 1
    config.truncated_backprop = True
    config.truncated_backprop_rand = True
    config.truncated_backprop_minmax = (35,49)
    config.trunc_backprop_timestep = 49
    config.prompt_fn = "simple_animals"
    config.reward_fn = "piexlate"
    config.eval_prompt_fn = "simple_animals"
    config.model_code = "image_pixelate_diff"
    config.reward_model_resume_from = "logs/2024.04.09_03.38.05/checkpoints/checkpoint_9/mlp.pt"
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 1
    config.train.learning_rate = 1e-3
    config.train.adam_weight_decay = 0
    config.save_freq = 2
    config.num_checkpoint_limit = 5    
    config.sd_guidance_scale = 7.5
    config.max_vis_images = 4
    config = set_config_batch(config,total_samples_per_epoch=256,total_batch_size= 64, per_gpu_capacity=2)
    return config

def img_compression_reward():
    config = general()
    config.wandb_name = "train_reward_for_img_compression"
    config.debug = False
    config.num_epochs = 120
    config.steps = 50
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "simple_animals"
    config.model_code = "image_compression_reward"
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 1
    config.train.learning_rate = 1e-3
    config.train.adam_weight_decay = 0
    config.save_freq = 20
    config.num_checkpoint_limit = 5    
    config.sd_guidance_scale = 7.5
    config.max_vis_images = 2
    config = set_config_batch(config,total_samples_per_epoch=256,total_batch_size= 64, per_gpu_capacity=8)
    return config

def img_compression_diff():
    config = general()
    config.wandb_name = "train_diff_for_img_compression"
    config.debug = True
    config.target_val = None
    config.num_epochs = 3
    config.steps = 50
    config.grad_scale = 1
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "simple_animals"
    config.model_code = "image_compression_diff"
    config.reward_model_resume_from = "logs/2024.04.05_00.18.35/checkpoints/checkpoint_2/mlp.pt"
    config.only_eval = False
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 1
    config.train.learning_rate = 1e-3
    config.train.adam_weight_decay = 0.01
    # config.resume_from = 'logs/usual-blaze-322/checkpoints'
    config.save_freq = 1
    config.num_checkpoint_limit = 10  
    config.sd_guidance_scale = 7.5  
    config = set_config_batch(config,total_samples_per_epoch=256,total_batch_size= 64, per_gpu_capacity=2)
    return config


def weather_reward():
    config = general()
    config.wandb_name = "train_reward_for_snow"
    config.debug = False
    config.num_epochs = 100
    config.steps = 50
    config.prompt_fn = "landscape"
    config.reward_fn = "snow"
    config.eval_prompt_fn = "landscape"
    config.model_code = "weather_reward"
    # config.resume_from = 'logs/2024.04.07_18.58.16/checkpoints/checkpoint_2/mlp.pt'
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 1
    config.train.learning_rate = 1e-3
    config.train.adam_weight_decay = 0
    config.save_freq = 4
    config.num_checkpoint_limit = 5    
    config.sd_guidance_scale = 7.5
    config.max_vis_images = 2
    config = set_config_batch(config,total_samples_per_epoch=256,total_batch_size= 64, per_gpu_capacity=4)
    return config

def weather_diff():
    config = general()
    config.wandb_name = "train_diff_for_snow"
    config.debug = False
    config.target_val = 0.8
    config.num_epochs = 100
    config.steps = 50
    config.grad_scale = 1
    config.prompt_fn = "landscape"
    config.reward_fn = "snow"
    config.eval_prompt_fn = "landscape"
    config.model_code = "weather_diff"
    config.reward_model_resume_from = "model_weight/snow.pt"
    config.only_eval = False
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 1
    config.train.learning_rate = 1e-3
    config.train.adam_weight_decay = 0.01
    # config.resume_from = 'logs/usual-blaze-322/checkpoints'
    config.save_freq = 1
    config.num_checkpoint_limit = 10  
    config.sd_guidance_scale = 7.5  
    config = set_config_batch(config,total_samples_per_epoch=256,total_batch_size= 64, per_gpu_capacity=2)
    return config

def get_config(name):
    return globals()[name]()