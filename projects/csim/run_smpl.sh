seqname=$1
dev=$2
fg_motion=urdf-smpl

logname2=fg-urdf
rm -rf logdir/$seqname-$logname2
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname2 \
  --fg_motion $fg_motion --num_rounds 20 --depth_wt 1e-3 --feature_type cse --freeze_scale --intrinsics_type const --init_scale_fg 0.5 --reg_pose_prior_wt 10 --nouse_freq_anneal --init_beta 0.01 --smpl_init --num_rounds_cam_init 1000 --reg_eikonal_wt 0.0

lognameft=$logname2-ft
rm -rf logdir/$seqname-$lognameft
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $lognameft \
  --fg_motion $fg_motion --num_rounds 120 --depth_wt 1e-3 --feature_type cse --freeze_scale --intrinsics_type const --init_scale_fg 0.5 --reg_pose_prior_wt 10 --nouse_freq_anneal \
  --load_path logdir/$seqname-$logname2/ckpt_latest.pth --nomlp_init --learning_rate 1e-4


logname1=bg
rm -rf logdir/$seqname-$logname1
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname1 \
  --extrinsics_type mixse3 --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 10000.0 \
  --field_type bg --data_prefix full --num_rounds 120 --mask_wt 0.01 \
  --depth_wt 0.01 --normal_wt 1e-3 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --freeze_scale --intrinsics_type const \
  --nouse_freq_anneal --init_scale_bg 0.2

logname=comp
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --pixels_per_image 12 \
  --field_type comp --fg_motion $fg_motion --num_rounds 20 --learning_rate 1e-4 --noreset_steps  --noabsorb_base \
  --load_path logdir/$seqname-$lognameft/ckpt_latest.pth --load_path_bg logdir/$seqname-$logname1/ckpt_latest.pth \
  --depth_wt 1e-3 --feat_reproj_wt 0.0 --feature_wt 0.0 --feature_type cse --freeze_scale --freeze_field_bg \
  --rgb_wt 1 --reg_eikonal_wt 0.1 --init_scale_bg 0.2