seqname=$1
field_type=$2 # fg
data_prefix=$3 # crop
dev=$4
batchsize=1
fg_motion=bob


logname=diffgs-fs-$field_type-b$batchsize-$fg_motion-20r
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu $batchsize --field_type $field_type --data_prefix $data_prefix --eval_res 256 \
  --num_rounds 20 --iters_per_round 200 --learning_rate 5e-3 \
  --feature_type cse --intrinsics_type const --extrinsics_type mlp --fg_motion $fg_motion \
  --use_init_cam --use_timesync \
  --reg_arap_wt 0.0 --num_pts 20000 --depth_wt 0.0 --num_workers 2 \
  --feature_wt 0 --xyz_wt 0 --extrinsics_type const
  # --extrinsics_type const
# python projects/diffgs/render.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
# python projects/diffgs/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full