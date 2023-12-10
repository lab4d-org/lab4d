# envname=Oct31at1-13AM-poly
envname=Oct5at10-49AM-poly
# envname=home-two
# vidname=2023-11-03--20-46-57
# vidname=cat-pikachu-6
# dev=2
vidname=$1
dev=$2

# # envname, dev
# bash projects/csim/recon_scene.sh $envname $dev

# run camera inference on the scene
python projects/predictor/inference.py --flagfile=logdir/predictor-comb-dino-rot-aug6-highres-b256-max-uniform-fixcrop2-img-ft4/opts.log \
  --load_suffix latest --image_dir database/processed/JPEGImages/Full-Resolution/$vidname-0000/
python projects/csim/transform_bg_cams.py $vidname-0000

# envname, actorname, dev
bash projects/csim/align_scene.sh $envname $vidname $dev

# foregroud from scratch with composition
seqname=home-$vidname
logname=compose-fs
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion urdf-quad \
  --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
  --freeze_scale --freeze_camera_bg --load_path_bg logdir/home-$vidname-bg-adapt3/ckpt_latest.pth --num_rounds 120 \
  --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-2 --reg_eikonal_wt 0.1 \
  --pixels_per_image 12 --bg_vid 0 \
  --nosingle_inst --beta_prob_init 0.0 --beta_prob_final 0.0 --noabsorb_base --reset_beta 0.01 --init_scale_fg 0.5

# fine-tune
seqname=home-$vidname
logname=compose-ft
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion urdf-quad \
  --intrinsics_type const --extrinsics_type mixse3 --feature_channels 384 \
  --freeze_scale --freeze_camera_bg --freeze_field_fgbg --learning_rate 1e-4 --noreset_steps --noabsorb_base --nouse_freq_anneal --num_rounds 20 \
  --load_path_bg logdir/home-$vidname-bg-adapt3/ckpt_latest.pth --load_path logdir/home-$vidname-compose-fs/ckpt_latest.pth \
  --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 \
  --pixels_per_image 12 --bg_vid 0 \
  --nosingle_inst --beta_prob_init 0.0 --beta_prob_final 0.0

CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --grid_size 256
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 1 --vis_thresh -10 --grid_size 256 --data_prefix full


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
#   # --data_prefix full
#   # --reg_cam_smooth_wt 0.01 --reg_cam_prior_wt 0.0 \