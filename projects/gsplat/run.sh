seqname=cat-pikachu-0
logname=gsplat-cat-z123
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/gsplat/train.py 0 --seqname $seqname --logname $logname --num_rounds 20 --iters_per_round 200 --learning_rate 5e-3 \
  --pixels_per_image -1 --imgs_per_gpu 1