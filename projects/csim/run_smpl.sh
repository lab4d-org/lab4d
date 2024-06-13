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