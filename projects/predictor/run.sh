seqname=predictor
dev=$1

#logname=Feb14at5-55тАпPM-poly
#logname=Feb26at10-02 PM-poly
logname=Feb19at9-47 PM-poly
rm -rf logdir/$seqname-$logname
rm -rf logdir/$seqname-$logname-ft1
rm -rf logdir/$seqname-$logname-ft2
rm -rf logdir/$seqname-$logname-ft3
rm -rf logdir/$seqname-$logname-ft4

rm -rf ~/.cache
ln -s /mnt/home/gengshany/.cache/ ~/.cache
rm -rf ~/.torch
ln -s /mnt/home/gengshany/.torch/ ~/.torch

bash scripts/train.sh projects/predictor/train.py $dev --seqname $seqname --logname $logname --num_rounds 40 --imgs_per_gpu 128 --iters_per_round 50 --poly_1 $logname --poly_2 $logname

bash scripts/train.sh projects/predictor/train.py $dev --seqname $seqname --logname $logname-ft1 --num_rounds 40 --imgs_per_gpu 128 --iters_per_round 50 \
    --load_path logdir/$seqname-$logname/ckpt_latest.pth --poly_1 $logname --poly_2 $logname

bash scripts/train.sh projects/predictor/train.py $dev --seqname $seqname --logname $logname-ft2 --num_rounds 40 --imgs_per_gpu 128 --iters_per_round 50 \
    --load_path logdir/$seqname-$logname-ft1/ckpt_latest.pth --poly_1 $logname --poly_2 $logname

bash scripts/train.sh projects/predictor/train.py $dev --seqname $seqname --logname $logname-ft3 --num_rounds 40 --imgs_per_gpu 128 --iters_per_round 50 \
    --load_path logdir/$seqname-$logname-ft2/ckpt_latest.pth --poly_1 $logname --poly_2 $logname

bash scripts/train.sh projects/predictor/train.py $dev --seqname $seqname --logname $logname-ft4 --num_rounds 40 --imgs_per_gpu 128 --iters_per_round 50 \
    --load_path logdir/$seqname-$logname-ft3/ckpt_latest.pth --poly_1 $logname --poly_2 $logname