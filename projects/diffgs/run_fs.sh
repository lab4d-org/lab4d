seqname=$1
lab4d_path=$2
field_type=$3 # fg
data_prefix=$4 # crop
dev=$5
batchsize=8


logname=diffgs-$field_type-b$batchsize
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu $batchsize --field_type $field_type --data_prefix $data_prefix --eval_res 256 \
  --num_rounds 120 --iters_per_round 200 --learning_rate 5e-3 \
  --feature_type cse --intrinsics_type const --extrinsics_type explicit --fg_motion bob \
  --use_init_cam --lab4d_path $lab4d_path --use_timesync \
  --reg_arap_wt 0.0 --num_pts 20000 --depth_wt 0.1
# python projects/gsplat/render.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
# python projects/gsplat/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
