# # dynamic
# dev=0
# seqname=eagle-d
# logname=gsplat-ref-test
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse

# rigid, no sds
dev=0
seqname=eagle
# seqname=cat-pikachu-0
logname=gsplat-ref-test3
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
  --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
  --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse

# # image-based
# dev=0
# # seqname=eagle
# seqname=cat-pikachu-0
# logname=gsplat-ref-z1234-5
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 120 --iters_per_round 50 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 2e-4 
#   # --rgb_wt 0.0 --mask_wt 0.0

# # text-based
# dev=0
# seqname=cat-pikachu-0
# logname=gsplat-text
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 10 --iters_per_round 50 --learning_rate 5e-3 \
#   --guidance_sd_wt 1e-4 --guidance_zero123_wt 0.0 --rgb_wt 0.0 --mask_wt 0.0