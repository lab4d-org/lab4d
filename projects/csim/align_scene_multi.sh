# align a target scene to canonical background scene
# input: record3d sequences
envname=$1
seqname=$2
dev=$3

# three stages training for bad initial alignment
# adapting: freeze field
logname=bg-adapt1
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
  --freeze_scale --freeze_field_bg --learning_rate 1e-4 --load_path logdir/$envname-bg/ckpt_latest.pth --nouse_freq_anneal --num_rounds 20 \
  --mask_wt 0.01 --normal_wt 0.0 --feature_wt 1e-3 --depth_wt 0.0 --reg_eikonal_wt 1e-3 --flow_wt 5e-4 --feat_reproj_wt 0.0 --rgb_wt 0.0 \
  --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 10000.0 \

# # adapting-v2: optimize field with the same loss
logname=bg-adapt2
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$seqname-bg-adapt1/ckpt_latest.pth --nouse_freq_anneal --num_rounds 20 \
  --mask_wt 0.01 --normal_wt 0.0 --feature_wt 1e-3 --depth_wt 0.0 --reg_eikonal_wt 1e-3 --flow_wt 5e-4 --feat_reproj_wt 0.0 --rgb_wt 0.0\
  --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0 \

# adapting-v3: add rgb loss
logname=bg-adapt3
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$seqname-bg-adapt2/ckpt_latest.pth --nouse_freq_anneal --num_rounds 80 \
  --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 --feat_reproj_wt 0.0 \
  --nosingle_inst --beta_prob_init 0.2 --beta_prob_final 0.0 --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0
  
# CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --grid_size 256
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 1 --vis_thresh -10 --grid_size 128
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 2 --vis_thresh -10 --grid_size 128
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 3 --vis_thresh -10 --grid_size 128
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 4 --vis_thresh -10 --grid_size 128
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 5 --vis_thresh -10 --grid_size 128
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 6 --vis_thresh -10 --grid_size 128
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 7 --vis_thresh -10 --grid_size 128
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 8 --vis_thresh -10 --grid_size 128
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 9 --vis_thresh -10 --grid_size 128