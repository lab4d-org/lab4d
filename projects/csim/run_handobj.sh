seqname=$1
dev=$2
fg_motion=bob

logname2=fg
rm -rf logdir/$seqname-$logname2
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname2 \
  --fg_motion $fg_motion --num_rounds 120 --depth_wt 1e-3 --feature_type cse --freeze_scale --intrinsics_type const --init_scale_fg 0.5 \
  --use_timesync  --reg_timesync_cam_wt 0.01 --reg_cam_smooth_wt 0.01 --nouse_cc

# logname1=bg
# rm -rf logdir/$seqname-$logname1
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname1 \
#   --extrinsics_type mixse3 --reg_cam_prior_wt 0.0 --reg_cam_prior_relative_wt 10000.0 \
#   --field_type bg --data_prefix full --num_rounds 120 --mask_wt 0.01 \
#   --depth_wt 0.01 --normal_wt 1e-3 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --freeze_scale --intrinsics_type const \
#   --nouse_freq_anneal --init_scale_bg 0.2 --flow_wt 0.0 --noupdate_geometry_aux --reg_visibility_wt 0.0