export OMP_NUM_THREADS=10
torchrun --nproc_per_node=2 train_ddp_stage2.py --model TinyLUTRE \
--expDir ../../models/ShiftLUT_F_int \
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
--cnum 16