# align a target scene to canonical background scene
# input: record3d sequences
envname=$1
seqname=$2
dev=$3
prefix=abs_noflow

# three stages training for bad initial alignment
# adapting: freeze field
logname=bg-adapt1-$prefix
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --freeze_field_bg --learning_rate 1e-4 --load_path logdir/$envname-bg/ckpt_latest.pth --nouse_freq_anneal --num_rounds 20 \
  --mask_wt 0.01 --normal_wt 0.0 --feature_wt 1e-3 --depth_wt 0.0 --reg_eikonal_wt 1e-3 --flow_wt 5e-4 --feat_reproj_wt 0.0 --rgb_wt 0.0 \
  --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 10000.0 \
  --imgs_per_gpu 512 \
  --flow_wt 0.0
  # --feature_channels 384

# # adapting-v2: optimize field with the same loss
logname=bg-adapt2-$prefix
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$seqname-bg-adapt1-$prefix/ckpt_latest.pth --nouse_freq_anneal --num_rounds 20 \
  --mask_wt 0.01 --normal_wt 0.0 --feature_wt 1e-3 --depth_wt 0.0 --reg_eikonal_wt 1e-3 --flow_wt 5e-4 --feat_reproj_wt 0.0 --rgb_wt 0.0\
  --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0 \
  --imgs_per_gpu 512 \
  --flow_wt 0.0
  # --feature_channels 384 

# adapting-v3: add rgb loss
logname=bg-adapt3-$prefix
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$seqname-bg-adapt2-$prefix/ckpt_latest.pth --nouse_freq_anneal --num_rounds 80 \
  --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 --feat_reproj_wt 0.0 \
  --nosingle_inst --beta_prob_init_bg 0.2 --beta_prob_final_bg 0.0 --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0 \
  --imgs_per_gpu 512 \
  --flow_wt 0.0
  # --feature_channels 384 