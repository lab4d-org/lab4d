vidname=$1
dev=$2

# # foregroud from scratch with composition
# seqname=home-$vidname
# logname=compose-fs
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion urdf-quad \
#   --intrinsics_type const --extrinsics_type mixse3 --feature_type cse \
#   --freeze_scale --freeze_camera_bg --load_path_bg logdir/home-$vidname-bg-adapt3/ckpt_latest.pth --num_rounds 120 \
#   --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-2 --reg_eikonal_wt 0.1 \
#   --pixels_per_image 12 --bg_vid 0 \
#   --nosingle_inst --beta_prob_init 0.0 --beta_prob_final 0.0 --noabsorb_base --reset_beta 0.01 --init_scale_fg 0.5

# # fine-tune
seqname=home-$vidname
logname=compose-ft
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion urdf-quad \
#   --intrinsics_type const --extrinsics_type mixse3 --feature_type cse \
#   --freeze_scale --freeze_camera_bg --freeze_field_fgbg --learning_rate 1e-4 --noreset_steps --noabsorb_base --nouse_freq_anneal --num_rounds 20 \
#   --load_path_bg logdir/home-$vidname-bg-adapt3/ckpt_latest.pth --load_path logdir/home-$vidname-compose-fs/ckpt_latest.pth \
#   --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-3 --reg_eikonal_wt 1e-3 \
#   --pixels_per_image 12 --bg_vid 0 \
#   --nosingle_inst --beta_prob_init 0.0 --beta_prob_final 0.0
# # --feature_channels 384 

# CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -20 --grid_size 128
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 1 --vis_thresh -20 --grid_size 128 --data_prefix full
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 2 --vis_thresh -20 --grid_size 128 --data_prefix full
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 3 --vis_thresh -20 --grid_size 128 --data_prefix full
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 4 --vis_thresh -20 --grid_size 128 --data_prefix full
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 5 --vis_thresh -20 --grid_size 128 --data_prefix full
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 6 --vis_thresh -20 --grid_size 128 --data_prefix full
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 7 --vis_thresh -20 --grid_size 128 --data_prefix full
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 8 --vis_thresh -20 --grid_size 128 --data_prefix full
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 9 --vis_thresh -20 --grid_size 128 --data_prefix full
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 10 --vis_thresh -20 --grid_size 128 --data_prefix full;
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 11 --vis_thresh -20 --grid_size 128 --data_prefix full;
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 12 --vis_thresh -20 --grid_size 128 --data_prefix full;
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 13 --vis_thresh -20 --grid_size 128 --data_prefix full;
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 14 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 15 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 16 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 17 --vis_thresh -20 --grid_size 128 --data_prefix full;
CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 18 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 19 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 20 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 21 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 22 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 23 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 24 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 25 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 26 --vis_thresh -20 --grid_size 128 --data_prefix full;
#CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 27 --vis_thresh -20 --grid_size 128 --data_prefix full;
#CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 28 --vis_thresh -20 --grid_size 128 --data_prefix full;
#CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 29 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 30 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 31 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 32 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 33 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 34 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 35 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 36 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 37 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 38 --vis_thresh -20 --grid_size 128 --data_prefix full;
#CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 39 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 40 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 41 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 42 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 43 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 44 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 45 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 46 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 47 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 48 --vis_thresh -20 --grid_size 128 --data_prefix full;
#CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 49 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 50 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 51 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 52 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 53 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 54 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 55 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 56 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 57 --vis_thresh -20 --grid_size 128 --data_prefix full;CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$seqname-$logname/opts.log --load_suffix latest --inst_id 58 --vis_thresh -20 --grid_size 128 --data_prefix full;


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