# pip install -e .
# NUM_GPUS=1
# export OPENAI_LOGDIR=debug

NUM_GPUS=8
export OPENAI_LOGDIR=movi-c-256_mdt_M2-bs24-beta-0.0195-lr1e-4-mr-0.5

MODEL_FLAGS="--image_size 256 --mask_ratio 0.50 --decode_layer 2 --model MDT_M_2 --save_interval 5000 --resume_checkpoint=movi-c-256_mdt_M2-bs24-beta-0.0195-lr1e-4-mr-0.5/model440000.pt"
# MODEL_FLAGS="--image_size 256 --mask_ratio 0.50 --decode_layer 2 --model MDT_M_2 --save_interval 5000"
# MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 2 --model MDT_L_4 --save_interval 5000"
# MODEL_FLAGS="--image_size 128 --mask_ratio 0.30 --decode_layer 2 --model MDT_S_2"
# MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 2 --model MDT_S_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 24"
#DATA_PATH=/dataset/imagenet
# DATA_PATH=/data2/common_datasets/movi_dataset/movi-c
DATA_PATH=/data2/common_datasets/movi_dataset/movi-c-256/movi-c-256

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
