seqname=$1
lab4d_path=$2
dev=$3

field_type=fg
data_prefix=crop
batchsize=16

logname=diffgs-$field_type-b$batchsize
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