# align a target scene to canonical background scene
# input: record3d sequences
envname=$1
seqname=$2
dev=$3

logname=bg-totalrecon-v2
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --learning_rate 1e-4 --load_path logdir/$envname-bg/ckpt_latest.pth --nouse_freq_anneal --num_rounds 80 \
  --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 --feat_reproj_wt 0.0 \
  --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 1.0 \
  --imgs_per_gpu 512 \
  --feature_wt 0.0 \
  --scene_type share-x --beta_prob_init_bg 0.0 --beta_prob_final_bg 0.0
  # --feature_channels 384 