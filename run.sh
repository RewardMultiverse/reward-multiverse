# CUDA_VISIBLE_DEVICES=0,1,3,4 accelerate launch train_reward.py --config config/all.py:img_pixelate_reward
CUDA_VISIBLE_DEVICES=0,1,3,4 accelerate launch train_diffusion.py --config config/all.py:img_pixelate_diff
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 nproc_per_node=1