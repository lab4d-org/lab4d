seqname=$1
lab4d_path=$2
field_type=$3 # fg
data_prefix=$4 # crop
dev=$5
batchsize=4
num_rounds=120

logname=diffgs-ft-$field_type-b$batchsize-r$num_rounds-sync
rm -rf logdir/$seqname-$logname
bash scripts/train.sh projects/diffgs/train.py $dev --seqname $seqname --logname $logname \
  --pixels_per_image -1 --imgs_per_gpu $batchsize --field_type $field_type --data_prefix $data_prefix --eval_res 256 \
  --num_rounds $num_rounds --iters_per_round 200 --learning_rate 5e-3 \
  --feature_type cse --intrinsics_type const --extrinsics_type const \
  --use_init_cam --lab4d_path $lab4d_path \
  --reg_arap_wt 0.0 --num_pts 20000 --depth_wt 0.01 \
  --feature_wt 0 --xyz_wt 0 --reg_gauss_skin_wt 0.0 --use_timesync
python projects/diffgs/render.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full
python projects/diffgs/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --data_prefix full