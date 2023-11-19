# reconstruct a canonical background scene
# input: polycam sequences
envname=$1
dev=$2

# single stage with frozen camera
seqname=$envname
logname=bg
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type const --feature_channels 384 \
  --freeze_scale --learning_rate 2e-3 --num_rounds 240 \
  --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 0.01 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --flow_wt 0.0
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$envname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --grid_size 256


# # initialize field with frozen camera
# seqname=$envname
# logname=bg-init
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
#   --intrinsics_type const --extrinsics_type const --feature_channels 384 \
#   --freeze_scale --learning_rate 1e-3 --num_rounds 20 \
#   --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 0.01 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --flow_wt 2e-3
# CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$envname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --grid_size 256

# # optimize camera as well
# seqname=$envname
# logname=bg-ft
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
#   --intrinsics_type const --feature_channels 384 \
#   --freeze_scale --learning_rate 1e-3 --num_rounds 120 --load_path logdir/$envname-bg-init/ckpt_latest.pth --noreset_steps --reset_beta \
#   --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 0.01 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --flow_wt 2e-3
# CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$envname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --grid_size 256