seqname=$1
field_type=$2 # fg
data_prefix=$3 # crop
dev=$4
batchsize=8
fg_motion=bob
# extrinsics_type=const # mlp
extrinsics_type=explicit # mlp
# gaussian_obj_scale=0.25 # hand
gaussian_obj_scale=0.5 # cat
num_rounds=120


logname=diffgs-fs-$field_type-b$batchsize-$fg_motion-r$num_rounds-mlp-fixgs-20reset-01th-rgbd001-exp-1m
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu $batchsize --field_type $field_type --data_prefix $data_prefix --eval_res 256 \
  --num_rounds $num_rounds --iters_per_round 200 --learning_rate 5e-3 \
  --feature_type cse --intrinsics_type const --extrinsics_type $extrinsics_type --fg_motion $fg_motion \
  --use_init_cam \
  --reg_arap_wt 0.0 --num_pts 100000 --depth_wt 0.01 --flow_wt 0.0 --mask_wt 0 --num_workers 2 \
  --feature_wt 0 --xyz_wt 0 --gaussian_obj_scale $gaussian_obj_scale # --use_timesync --align_root_pose_from_bgcam --reg_timesync_cam_wt 0.01 # 
# python projects/diffgs/render.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
# python projects/diffgs/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
