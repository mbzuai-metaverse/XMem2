STAGE=0
B=6 # original 16
LR=3.75e-06 # original 1e-5 / (16/6) batch size difference 
START_WARM=53000 # original 20K iterations * (16/6) batch size difference 
END_WARM=186000 # original 70K iterations * (16/6) batch size difference
NUM_ITER=400000 # original 150K iterations * (16/6) batch size difference 
NUM_WORKERS=6

OMP_NUM_THREADS=6 python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 train.py \
    --exp_id xmem_multiscale \
    --stage "$STAGE" \
    --s0_batch_size="$B" \
    --s0_lr="$LR" \
    --s0_start_warm="$START_WARM" \
    --s0_end_warm="$END_WARM" \
    --s0_iterations="$NUM_ITER" \
    --num_workers="$NUM_WORKERS"