export OMP_NUM_THREADS=10
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29510 train_ddp_stage2.py --model TinyLUTRE \
--expDir ../../models/ShiftLUT_F_denoising_15_int \
--trainDir /user/train/DIV2K \
--valDir /user/val \
--batchSize 16 \
--lr0 5e-3 \
--lr1 5e-4 \
--stack 7 \
--workerNum 0 \
--totalIter 200000 \
--use_shift \
--valStep 1000 \
--saveStep 1000 \
--msb_base 6 \
--cnum 16 \
--sigma 15 \
--scale 1