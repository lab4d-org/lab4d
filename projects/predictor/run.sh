# seqname=predictor

# logname=comb-dino-rot-aug6-highres-b256-max-uniform-fixcrop2-img
# # rm -rf logdir/$seqname-$logname
# rm -rf logdir/$seqname-$logname-ft1
# rm -rf logdir/$seqname-$logname-ft2
# rm -rf logdir/$seqname-$logname-ft3
# rm -rf logdir/$seqname-$logname-ft4
# # bash scripts/train.sh projects/predictor/train.py 1,2 --seqname $seqname --logname $logname --num_rounds 30 --imgs_per_gpu 128 --iters_per_round 50

# bash scripts/train.sh projects/predictor/train.py 1,2 --seqname $seqname --logname $logname-ft1 --num_rounds 30 --imgs_per_gpu 128 --iters_per_round 50 \
#     --load_path logdir/$seqname-$logname/ckpt_latest.pth

# bash scripts/train.sh projects/predictor/train.py 1,2 --seqname $seqname --logname $logname-ft2 --num_rounds 30 --imgs_per_gpu 128 --iters_per_round 50 \
#     --load_path logdir/$seqname-$logname-ft1/ckpt_latest.pth

# bash scripts/train.sh projects/predictor/train.py 1,2 --seqname $seqname --logname $logname-ft3 --num_rounds 30 --imgs_per_gpu 128 --iters_per_round 50 \
#     --load_path logdir/$seqname-$logname-ft2/ckpt_latest.pth

# bash scripts/train.sh projects/predictor/train.py 1,2 --seqname $seqname --logname $logname-ft4 --num_rounds 30 --imgs_per_gpu 128 --iters_per_round 50 \
#     --load_path logdir/$seqname-$logname-ft3/ckpt_latest.pth