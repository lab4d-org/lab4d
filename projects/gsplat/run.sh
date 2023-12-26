# rigid singlecam incremental
dev=1
seqname=eagle-s-0000
logname=gsplat-ref
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
  --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
  --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse --use_timesync \
  --intrinsics_type const --extrinsics_type const --fg_motion dynamic \
  --reg_least_action_wt 0.0 --reg_arap_wt 0.1 --inc_warmup_ratio 0.25
# --guidance_zero123_wt 2e-4

# # rigid multiview incremental
# dev=1
# seqname=eagle-s
# logname=gsplat-ref-inc-full-auto2
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse --use_timesync \
#   --intrinsics_type const --extrinsics_type const --fg_motion dynamic \
#   --reg_least_action_wt 0.0 --reg_arap_wt 0.1 --inc_warmup_ratio 0.25

# # rigid multiview
# dev=0
# seqname=eagle-s
# logname=gsplat-ref-01arap-const-5k-cuda2k-global
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse --use_timesync \
#   --intrinsics_type const --extrinsics_type const --fg_motion dynamic \
#   --reg_least_action_wt 0.0 --reg_arap_wt 0.1


# # dynamic
# dev=0
# seqname=eagle-d
# logname=gsplat-ref-rot-arap
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse --use_timesync \
#   --intrinsics_type const --extrinsics_type const --fg_motion dynamic \
#   --reg_least_action_wt 0.0 --reg_arap_wt 0.1

# # rigid, no sds
# dev=0
# seqname=eagle
# # seqname=cat-pikachu-0
# logname=gsplat-ref-resetstat
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse --fg_motion rigid

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