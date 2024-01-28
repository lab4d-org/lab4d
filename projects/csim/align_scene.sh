# align a target scene to canonical background scene
# input: record3d sequences
envname=$1
vidname=$2
dev=$3

# ## single stage training for good initial alignment (needs to freeze background and adjust camera at the begining)
# ## TODO: joint training in a single stage
# seqname=home-$vidname
# logname=bg
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
#   --intrinsics_type const --extrinsics_type mix --feature_channels 384 \
#   --freeze_scale --learning_rate 1e-4 --load_path logdir/$envname-bg/ckpt_latest.pth --nouse_freq_anneal --num_rounds 120 \
#   --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 0.01 --reg_eikonal_wt 1e-3 --flow_wt 2e-3 --feat_reproj_wt 0.0 \
#   --nosingle_inst --beta_prob_final 0.0 --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0
# CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 1 --vis_thresh -10 --grid_size 256

# three stages training for bad initial alignment
# adapting: freeze field
seqname=home-$vidname
logname=bg-adapt1
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
  --freeze_scale --freeze_field_bg --learning_rate 1e-4 --load_path logdir/$envname-bg/ckpt_latest.pth --nouse_freq_anneal --num_rounds 20 \
  --mask_wt 0.01 --normal_wt 0.0 --feature_wt 1e-3 --depth_wt 0.0 --reg_eikonal_wt 1e-3 --flow_wt 5e-4 --feat_reproj_wt 0.0 --rgb_wt 0.0 \
  --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 10000.0 \
  # --init_scale_bg 0.2 
  # --reg_cam_smooth_wt 0.01

# adapting-v2: optimize field with the same loss
seqname=home-$vidname
logname=bg-adapt2
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$seqname-bg-adapt1/ckpt_latest.pth --nouse_freq_anneal --num_rounds 20 \
  --mask_wt 0.01 --normal_wt 0.0 --feature_wt 1e-3 --depth_wt 0.0 --reg_eikonal_wt 1e-3 --flow_wt 5e-4 --feat_reproj_wt 0.0 --rgb_wt 0.0\
  --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0 \
  # --init_scale_bg 0.2 
  # --reg_cam_smooth_wt 0.01

# adapting-v3: add rgb loss
seqname=home-$vidname
logname=bg-adapt3
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$seqname-bg-adapt2/ckpt_latest.pth --nouse_freq_anneal --num_rounds 80 \
  --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 --feat_reproj_wt 0.0 \
  --nosingle_inst --beta_prob_init 0.2 --beta_prob_final 0.0 --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0
  # --init_scale_bg 0.2 
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --grid_size 256
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 1 --vis_thresh -10 --grid_size 256