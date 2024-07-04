seqname=$1
lab4d_path=$2
dev=$3

field_type=comp # fg # bg
data_prefix=full # crop
batchsize=16

# dynamic singlecam
# dev=0
#seqname=cat-pikachu-0
# seqname=home-2023-curated3
# logname=gsplat-ref-lab4d-shadowrgb-opt-sync
logname=diffgs-$field_type-b$batchsize
# lab4d_path=logdir/home-2023-curated3-compose-ft/opts.log
# lab4d_path=logdir/cat-pikachu-0-fg-skel/opts.log
# lab4d_path=logdir/cat-pikachu-0-comp/opts.log
# lab4d_path=logdir/2024-05-07--19-25-33-fg-urdf-sync-fix-sm-ft/opts.log
# lab4d_path=logdir/Oct5at10-49AM-poly-bg/opts.log
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu $batchsize --field_type $field_type --data_prefix $data_prefix --eval_res 256 \
  --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
  --feature_type cse --intrinsics_type const --extrinsics_type explicit --fg_motion bob \
  --use_init_cam --lab4d_path $lab4d_path --use_timesync \
  --reg_arap_wt 0.0 --num_pts 20000 --depth_wt 0.1
  # --depth_wt 0.01 --flow_wt 0.1
  # --flow_wt 0 --depth_wt 0.1
  # --load_path logdir/$seqname-gsplat-ref-lab4d-comp3/ckpt_latest.pth
  # --flow_wt 0.0 \
  # --bg_vid 0 # --guidance_zero123_wt 2e-4
  # --flow_wt 0.1 --reg_arap_wt 1.0 \
  # --extrinsics_type image --fg_motion image --reg_lab4d_wt 1.0
# python projects/gsplat/render.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
# python projects/gsplat/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full

# # dynamic singlecam incremental
# dev=2
# seqname=cat-pikachu-0
# # seqname=eagle-s-0001
# # seqname=2023-11-03--20-53-19
# logname=gsplat-ref-z123
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 0 --iters_per_round 200 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse --use_timesync \
#   --intrinsics_type const --extrinsics_type explicit --fg_motion dynamic \
#   --reg_arap_wt 1.0 --inc_warmup_ratio 1.0 --flow_wt 0.1 --num_pts 500 --delta_list ","\
#   --guidance_zero123_wt 2e-4 --save_freq 10

# # rigid singlecam incremental
# dev=1
# seqname=eagle-s-0001
# logname=gsplat-ref-flow-arap-opt2-alldata-z123
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse --use_timesync \
#   --intrinsics_type const --extrinsics_type const --fg_motion dynamic \
#   --reg_arap_wt 1.0 --inc_warmup_ratio 0.25 --flow_wt 1.0 --num_pts 500 \
#   --guidance_zero123_wt 5e-4

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


# # dynamic multiview
# dev=1
# seqname=eagle-d
# logname=gsplat-ref
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --intrinsics_type const --extrinsics_type const --fg_motion dynamic \
#   --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 --eval_res 256 \
#   --feature_type cse --use_timesync \
#   --reg_arap_wt 1.0 --flow_wt 1.0

# # rigid, no sds
# dev=0
# seqname=eagle
# # seqname=cat-pikachu-0
# logname=gsplat-ref
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg --intrinsics_type const --extrinsics_type const \
#   --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
#   --guidance_sd_wt 0.0 --guidance_zero123_wt 0.0 --feature_type cse --fg_motion rigid --eval_res 256 --flow_wt 0.0

# # image-based
# dev=1
# # seqname=eagle-s-0001
# seqname=cat-pikachu-0
# logname=gsplat-ref
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg --intrinsics_type const --extrinsics_type const \
#   --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 --eval_res 256 --feature_type cse --fg_motion dynamic \
#   --guidance_zero123_wt 2e-4 --flow_wt 1.0 --reg_arap_wt 1.0
#   # --guidance_zero123_wt 2e-4 --flow_wt 0.0 --reg_arap_wt 0.0 --mask_wt 0.0 --rgb_wt 0.0
#   # --guidance_zero123_wt 2e-4 --flow_wt 0.0 --reg_arap_wt 0.0
#   # --guidance_zero123_wt 0 --flow_wt 0.0 --reg_arap_wt 0.0

# # text-based
# dev=0
# seqname=cat-pikachu-0
# logname=gsplat-text
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/gsplat/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu 1 --field_type fg \
#   --num_rounds 10 --iters_per_round 50 --learning_rate 5e-3 \
#   --guidance_sd_wt 1e-4 --guidance_zero123_wt 0.0 --rgb_wt 0.0 --mask_wt 0.0