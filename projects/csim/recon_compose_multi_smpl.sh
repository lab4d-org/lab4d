seqname=$1
fg_motion=urdf-smpl
dev=$2

# foregroud from scratch with composition
# # smpl
logname2=fg-urdf
# rm -rf logdir/$seqname-$logname2
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname2 \
#   --fg_motion $fg_motion --num_rounds 20 --depth_wt 1e-3 --feature_type cse --freeze_scale --intrinsics_type const --init_scale_fg 0.5 --reg_pose_prior_wt 10 --nouse_freq_anneal --init_beta 0.01 --smpl_init --num_rounds_cam_init 1000 --reg_eikonal_wt 0.0 \
#   --extrinsics_type mixse3 --bg_vid 0 

lognameft=$logname2-ft
# rm -rf logdir/$seqname-$lognameft
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $lognameft \
#   --fg_motion $fg_motion --num_rounds 120 --depth_wt 1e-3 --feature_type cse --freeze_scale --intrinsics_type const --init_scale_fg 0.5 --reg_pose_prior_wt 10 --nouse_freq_anneal \
#   --load_path logdir/$seqname-$logname2/ckpt_latest.pth --nomlp_init --learning_rate 1e-4 \
#   --extrinsics_type mixse3 --bg_vid 0 

# foregroud from scratch with composition
logname=compose-fs
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion $fg_motion \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --freeze_camera_bg --load_path_bg logdir/$seqname-bg-adapt3/ckpt_latest.pth --num_rounds 120 \
  --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-2 --reg_eikonal_wt 0.1 \
  --pixels_per_image 12 --bg_vid 0 \
  --scene_type share-x --beta_prob_init_bg 0.0 --beta_prob_final_bg 0.0 --beta_prob_init_fg 1.0 --beta_prob_final_fg 1.0 --noabsorb_base --init_scale_fg 0.5 \
  --imgs_per_gpu 512 --feature_type cse \
  --load_path logdir/$seqname-$lognameft/ckpt_latest.pth --nouse_freq_anneal --freeze_field_fg \
  --reg_pose_prior_wt 10 --learning_rate 1e-4
  # --noload_fg_camera  --reg_cam_smooth_wt 0.01 --reset_beta 0.01\

# fine-tune
logname=compose-ft
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion $fg_motion \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --freeze_camera_bg --freeze_field_fgbg --learning_rate 1e-4 --noreset_steps --noabsorb_base --nouse_freq_anneal --num_rounds 20 \
  --load_path_bg logdir/$seqname-bg-adapt3/ckpt_latest.pth --load_path logdir/$seqname-compose-fs/ckpt_latest.pth \
  --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 \
  --pixels_per_image 12 --bg_vid 0 \
  --scene_type share-x --beta_prob_init_bg 0.0 --beta_prob_final_bg 0.0 --beta_prob_init_fg 1.0 --beta_prob_final_fg 1.0 \
  --imgs_per_gpu 512 --feature_type cse --nomlp_init