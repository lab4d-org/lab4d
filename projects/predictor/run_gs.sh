seqname=predictor
inside_out=no
folder_name=$1
logname=$2
dev=$3

rm -rf logdir/$seqname-$logname

rm -rf ~/.cache
ln -s /mnt/home/gengshany/.cache/ ~/.cache
rm -rf ~/.torch
ln -s /mnt/home/gengshany/.torch/ ~/.torch

bash scripts/train.sh projects/predictor/train.py $dev --seqname $seqname --logname $logname --num_rounds 40 --imgs_per_gpu 128 --iters_per_round 50