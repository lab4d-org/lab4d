# seqname=2023-11-08--20-29-39
seqname=2023-11-11--11-54-06
dev=2

# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname bg \
#   --field_type bg --data_prefix full --num_rounds 120 --alter_flow --mask_wt 0.01 \
#   --normal_wt 1e-2 --reg_eikonal_wt 0.001 --depth_wt 0.01 --feature_channels 384 --freeze_scale --feat_reproj_wt 0.0 

# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname fg-urdf \
#   --fg_motion urdf-quad --num_rounds 120 --depth_wt 1e-2 --feature_channels 384 --freeze_scale --feat_reproj_wt 0.0 

logdir=logdir/$seqname-comp
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname comp --pixels_per_image 12 \
#   --field_type comp --fg_motion urdf-quad --num_rounds 20 --learning_rate 1e-4 --noreset_steps  --noabsorb_base \
#   --load_path logdir/$seqname-fg-urdf/ckpt_latest.pth --load_path_bg logdir/$seqname-bg/ckpt_latest.pth \
#   --depth_wt 1e-2 --feature_channels 384 --freeze_scale --feat_reproj_wt 0.0


# visualization
python lab4d/render_intermediate.py --testdir logdir/$seqname-ppr/ --data_class sim
python lab4d/export.py --flagfile=$logdir/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --extend_aabb
python lab4d/render_mesh.py --testdir $logdir/export_0000/ --view bev --ghosting
