seqname=$1
fg_motion=$2
dev=$3

# foregroud from scratch with composition
logname=compose-fs
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion $fg_motion \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --freeze_camera_bg --load_path_bg logdir/$seqname-bg-adapt3/ckpt_latest.pth --num_rounds 120 \
  --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-2 --reg_eikonal_wt 0.1 \
  --pixels_per_image 12 --bg_vid 0 \
  --scene_type share-x --beta_prob_init_bg 0.0 --beta_prob_final_bg 0.0 --beta_prob_init_fg 1.0 --beta_prob_final_fg 1.0 --noabsorb_base --reset_beta 0.01 --init_scale_fg 0.5 \
  --imgs_per_gpu 512 --feature_type cse \
  # --load_path logdir/home-2023-curated3-compose-ft/ckpt_latest.pth --nouse_freq_anneal --noload_fg_camera --freeze_field_fg \
  # --load_path logdir-old/home-2023-11-curated-compose-ft/ckpt_latest.pth --nouse_freq_anneal --noload_fg_camera --freeze_field_fg \

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
  --imgs_per_gpu 512 --feature_type cse



  ## old ones

# # actorname, dev
# bash projects/csim/recon_actor.sh $vidname $dev

# # compose: (1) freeze cameras  (2) single field (3) low depth weight (4) remove fg renderings on bg scene (5) no feature weight
# seqname=home-$vidname
# logname=compose
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion urdf-quad \
#   --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
#   --freeze_scale --freeze_field_fgbg --freeze_camera_bg --learning_rate 1e-4 --load_path logdir/$vidname-fg-urdf/ckpt_latest.pth  --load_path_bg logdir/home-$vidname-bg-adapt3/ckpt_latest.pth --noreset_steps --noabsorb_base --nouse_freq_anneal --num_rounds 20 \
#   --mask_wt 0.1 --normal_wt 0.0 --feature_wt 0.0 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 --flow_wt 0.0 --feat_reproj_wt 0.0 \
#   --pixels_per_image 12 --bg_vid 0 --reg_delta_skin_wt 0.0 --reg_skel_prior_wt 0.0 \
#   --nosingle_inst --beta_prob_init 0.0 --beta_prob_final 0.0
#   #  --reg_gauss_mask_wt 0.0 --freeze_camera_fg --init_scale_bg 0.2 

# # ppr
# seqname=home-$vidname
# logname=ppr
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh projects/ppr/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion urdf-quad \
#   --iters_per_round 100 --secs_per_wdw 2.4 --phys_vid 1 \
#   --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
#   --freeze_scale --learning_rate 1e-4 --load_path logdir/$vidname-fg-urdf/ckpt_latest.pth  --load_path_bg logdir/home-$vidname-bg-adapt3/ckpt_latest.pth --noreset_steps --noabsorb_base --nouse_freq_anneal --num_rounds 20 \
#   --mask_wt 0.1 --normal_wt 0.0 --feature_wt 0.0 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 --flow_wt 0.0 --feat_reproj_wt 0.0 \
#   --pixels_per_image 12 --bg_vid 0 --reg_delta_skin_wt 0.0 --reg_skel_prior_wt 0.0 \
#   --nosingle_inst --beta_prob_final 0.0
#   # --freeze_field_fgbg --freeze_camera_bg --freeze_camera_fg
#   # --nosingle_scene 
#   # --data_prefix full 0
#   # --reg_cam_smooth_wt 0.01 --reg_cam_prior_wt 0.0 \