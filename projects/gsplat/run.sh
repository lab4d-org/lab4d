# no sds
dev=0
seqname=cat-pikachu-0
logname=gsplat-cat-ref
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
  --num_rounds 10 --iters_per_round 50 --learning_rate 5e-3 \
  --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0

# # image-based
# dev=0
# seqname=cat-pikachu-0
# logname=gsplat-cat-ref-z1234
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 10 --iters_per_round 50 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 2e-4 

# # text-based
# dev=0
# seqname=cat-pikachu-0
# logname=gsplat-text
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 10 --iters_per_round 50 --learning_rate 5e-3 \
#   --guidance_sd_wt 1e-4 --guidance_zero123_wt 0.0 --rgb_wt 0.0 --mask_wt 0.0