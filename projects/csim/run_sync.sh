# seqname=2023-11-08--20-29-39
#seqname=2023-11-11--11-54-06
#dev=2
seqname=$1
dev=$2

logname2=fg-urdf-sync-fix
rm -rf logdir/$seqname-$logname2
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname2 \
  --fg_motion urdf-quad --num_rounds 120 --depth_wt 1e-3 --feature_type cse --freeze_scale --intrinsics_type const --init_scale_fg 0.4 \
  --use_timesync  --reg_timesync_cam_wt 0.01

# use higher rgb weight
logname3=fg-urdf-sync-fix-ft
rm -rf logdir/$seqname-$logname3
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname3 \
  --fg_motion urdf-quad --num_rounds 20 --learning_rate 1e-4 --noreset_steps  --noabsorb_base \
  --load_path logdir/$seqname-$logname2/ckpt_latest.pth \
  --depth_wt 1e-3 --feature_type cse --freeze_scale --intrinsics_type const \
  --use_timesync --rgb_wt 1 --reg_eikonal_wt 0.1 \

# logname1=bg
# rm -rf logdir/$seqname-$logname1
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname1 \
#   --extrinsics_type mixse3 --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 10000.0 \
#   --field_type bg --data_prefix full --num_rounds 120 --alter_flow --mask_wt 0.01 \
#   --depth_wt 1e-3 --normal_wt 1e-2 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --freeze_scale --intrinsics_type const \

# logname=comp
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --pixels_per_image 12 \
#   --field_type comp --fg_motion urdf-quad --num_rounds 20 --learning_rate 1e-4 --noreset_steps  --noabsorb_base \
#   --load_path logdir/$seqname-$logname2/ckpt_latest.pth --load_path_bg logdir/$seqname-$logname1/ckpt_latest.pth \
#   --depth_wt 1e-3 --feat_reproj_wt 0.0 --feature_wt 0.0 --feature_type cse --freeze_scale --freeze_field_bg \
#   --use_timesync

# logname=ppr
# bash scripts/train.sh projects/ppr/train.py $dev --seqname $seqname --logname $logname --pixels_per_image 12 \
#   --field_type comp --fg_motion urdf-quad --num_rounds 20 --learning_rate 1e-4 --noreset_steps  --noabsorb_base \
#   --load_path logdir/$seqname-$logname/ckpt_latest.pth \
#   --depth_wt 1e-2 --feature_channels 384 --freeze_scale --feat_reproj_wt 0.0 \
#   --iters_per_round 100 --secs_per_wdw 2.4


# logdir=logdir/$seqname-$logname
# # visualization
# python projects/ppr/render_intermediate.py --testdir $logdir --data_class sim
# python projects/ppr/export.py --flagfile=$logdir/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --extend_aabb
# python lab4d/render_mesh.py --testdir $logdir/export_0000/ --view bev --ghosting
