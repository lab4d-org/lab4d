seqname=$1
fg_motion=$2
dev=$3

# # foregroud from scratch with composition
# logname=compose-fs
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion $fg_motion \
#   --intrinsics_type const --extrinsics_type mixse3 \
#   --freeze_scale --freeze_camera_bg --load_path_bg logdir/$seqname-bg-adapt3/ckpt_latest.pth --num_rounds 120 \
#   --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-2 --reg_eikonal_wt 0.1 \
#   --pixels_per_image 12 --bg_vid 0 \
#   --scene_type share-x --beta_prob_init_bg 0.0 --beta_prob_final_bg 0.0 --beta_prob_init_fg 1.0 --beta_prob_final_fg 1.0 --noabsorb_base --reset_beta 0.01 --init_scale_fg 0.5 \
#   --imgs_per_gpu 512 --feature_type cse

# fine-tune
logname=compose-ft
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion $fg_motion \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --freeze_camera_bg --freeze_field_bg --learning_rate 1e-4 --noreset_steps --noabsorb_base --nouse_freq_anneal --num_rounds 20 \
  --load_path_bg logdir/$seqname-bg-adapt4/ckpt_latest.pth --load_path logdir/$seqname-compose-fs/ckpt_latest.pth \
  --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-2 --reg_eikonal_wt 1e-3 \
  --pixels_per_image 12 --bg_vid 0 \
  --scene_type share-x --beta_prob_init_bg 0.0 --beta_prob_final_bg 0.0 --beta_prob_init_fg 1.0 --beta_prob_final_fg 1.0 \
  --imgs_per_gpu 128 --feature_type cse --nomlp_init --n_depth 256 --rgb_wt 0.5
