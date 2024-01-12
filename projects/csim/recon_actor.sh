vidname=$1
dev=$2

# object
seqname=$vidname
logname=fg-urdf-4x-symmb-sdf
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --fg_motion urdf-quad --intrinsics_type const --num_rounds 120 --depth_wt 1e-3 \
  --imgs_per_gpu 512 --symm_ratio 1.0  --feature_type cse \
# --fg_motion urdf-human --use_timesync
# --feature_channels 384
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$vidname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh 0 --grid_size 256