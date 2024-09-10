# align a target scene to canonical background scene
# input: record3d sequences
envname=$1
seqname=$2
dev=$3
imgs_per_gpu=256 # original: 512

# three stages training for bad initial alignment
# adapting: freeze field
logname=bg-adapt1
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --freeze_field_bg --learning_rate 1e-4 --load_path logdir/$envname-bg/ckpt_latest.pth --nouse_freq_anneal --num_rounds 20 \
  --mask_wt 0.01 --normal_wt 0.0 --feature_wt 1e-3 --depth_wt 0.0 --reg_eikonal_wt 1e-3 --flow_wt 5e-4 --feat_reproj_wt 0.0 --rgb_wt 0.0 \
  --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 10000.0 \
  --imgs_per_gpu $imgs_per_gpu
  # --feature_channels 384

# # adapting-v2: optimize field with the same loss
logname=bg-adapt2
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$seqname-bg-adapt1/ckpt_latest.pth --nouse_freq_anneal --num_rounds 20 \
  --mask_wt 0.01 --normal_wt 0.0 --feature_wt 1e-3 --depth_wt 0.0 --reg_eikonal_wt 1e-3 --flow_wt 5e-4 --feat_reproj_wt 0.0 --rgb_wt 0.0\
  --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0 \
  --imgs_per_gpu $imgs_per_gpu
  # --feature_channels 384 

# adapting-v3: add rgb loss
logname=bg-adapt3
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$seqname-bg-adapt2/ckpt_latest.pth --nouse_freq_anneal --num_rounds 80 \
  --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 --feat_reproj_wt 0.0 \
  --scene_type share-x --beta_prob_init_bg 1.0 --beta_prob_final_bg 0.05 --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0 \
  --imgs_per_gpu $imgs_per_gpu
  # --feature_channels 384 

# adapting-v4: 10x depth loss, also n_depth=256
logname=bg-adapt4
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$seqname-bg-adapt3/ckpt_latest.pth --nouse_freq_anneal --num_rounds 20 \
  --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 1e-2 --reg_eikonal_wt 1e-3 --feat_reproj_wt 0.0 \
  --scene_type share-x --beta_prob_init_bg 0.05 --beta_prob_final_bg 0.05 --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0 \
  --imgs_per_gpu $((imgs_per_gpu/4)) --n_depth 256 --rgb_wt 0.5 --freeze_camera_bg
  # --feature_channels 384 