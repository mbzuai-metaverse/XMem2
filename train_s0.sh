STAGE=0
B=4 # original 16
LR=2.5e-06 # original 1e-5 / (16/4) batch size difference 
START_WARM=80000 # original 20K iterations * (16/4) batch size difference 
END_WARM=280000 # original 70K iterations * (16/4) batch size difference
NUM_ITER=600000 # original 150K iterations * (16/4) batch size difference 
NUM_WORKERS=6

OMP_NUM_THREADS=6 python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 train.py \
    --exp_id xmem_multiscale \
    --stage "$STAGE" \
    --s0_batch_size="$B" \
    --s0_lr="$LR" \
    --s0_start_warm="$START_WARM" \
    --s0_end_warm="$END_WARM" \
    --s0_iterations="$NUM_ITER" \
    --num_workers="$NUM_WORKERS" \ 
    --load_checkpoint='last'
