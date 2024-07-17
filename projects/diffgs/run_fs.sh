seqname=$1
field_type=$2 # fg
data_prefix=$3 # crop
dev=$4
batchsize=32
fg_motion=bob
gaussian_obj_scale=0.25 # hand
# gaussian_obj_scale=0.5 # cat
num_rounds=120


logname=diffgs-fs-$field_type-b$batchsize-$fg_motion-r$num_rounds-mlp
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu $batchsize --field_type $field_type --data_prefix $data_prefix --eval_res 256 \
  --num_rounds $num_rounds --iters_per_round 200 --learning_rate 5e-3 \
  --feature_type cse --intrinsics_type const --extrinsics_type mlp --fg_motion $fg_motion \
  --use_init_cam --use_timesync \
  --reg_arap_wt 0.0 --num_pts 20000 --depth_wt 0.0 --num_workers 2 \
  --feature_wt 0 --xyz_wt 0 --extrinsics_type mlp --gaussian_obj_scale $gaussian_obj_scale --align_root_pose_from_bgcam --reg_timesync_cam_wt 0.01 # --depth_wt 0.1 

# # init
# batchsize=1
# logname=diffgs-fs-$field_type-b$batchsize-$fg_motion-20r-mlp-align-init
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname \
#   --pixels_per_image -1 --imgs_per_gpu $batchsize --field_type $field_type --data_prefix $data_prefix --eval_res 256 \
#   --num_rounds 10 --iters_per_round 200 --learning_rate 5e-3 \
#   --feature_type cse --intrinsics_type const --extrinsics_type mlp --fg_motion $fg_motion \
#   --use_init_cam --use_timesync \
#   --reg_arap_wt 0.0 --num_pts 20000 --depth_wt 0.0 --num_workers 2 \
#   --feature_wt 0 --xyz_wt 0 --extrinsics_type mlp --reg_timesync_cam_wt 0.01 --gaussian_obj_scale $gaussian_obj_scale # --depth_wt 0.1 # --align_root_pose_from_bgcam 
#   # --extrinsics_type const

# fine-tune
# batchsize=24
# logname_ft=diffgs-fs-$field_type-b$batchsize-$fg_motion-20r-mlp-align-ft
# rm -rf logdir/$seqname-$logname_ft
# bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname_ft \
#   --pixels_per_image -1 --imgs_per_gpu $batchsize --field_type $field_type --data_prefix $data_prefix --eval_res 256 \
#   --num_rounds 20 --iters_per_round 200 --learning_rate 5e-3 \
#   --feature_type cse --intrinsics_type const --extrinsics_type mlp --fg_motion $fg_motion \
#   --use_init_cam --use_timesync \
#   --reg_arap_wt 0.0 --num_pts 20000 --depth_wt 0.0 --num_workers 2 \
#   --feature_wt 0 --xyz_wt 0 --extrinsics_type mlp --reg_timesync_cam_wt 0.01 --gaussian_obj_scale $gaussian_obj_scale --load_path logdir/$seqname-$logname/ckpt_latest.pth # --depth_wt 0.1 # --align_root_pose_from_bgcam 
# python projects/diffgs/render.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
# python projects/diffgs/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
