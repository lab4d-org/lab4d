seqname=$1
dev=$2
fg_motion=bob

logname2=$fg_motion
rm -rf logdir/$seqname-$logname2
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname2 \
  --fg_motion $fg_motion --num_rounds 20 --feature_type cse --intrinsics_type const --init_scale_fg 0.5 --imgs_per_gpu 128 \
  --use_timesync # --reg_timesync_cam_wt 0.01 # --depth_wt 1e-3
#   --fg_motion $fg_motion --num_rounds 20 --feature_type cse --intrinsics_type const --init_scale_fg 0.1 --extrinsics_type mlp_nodelta --imgs_per_gpu 128 \

# # use higher rgb weight
# logname3=$logname2-ft
# rm -rf logdir/$seqname-$logname3
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname3 \
#   --fg_motion urdf-quad --num_rounds 20 --learning_rate 1e-4 --noreset_steps  --noabsorb_base \
#   --load_path logdir/$seqname-$logname2/ckpt_latest.pth \
#   --depth_wt 1e-3 --feature_type cse --freeze_scale --intrinsics_type const \
#   --use_timesync --rgb_wt 1 --reg_eikonal_wt 0.1 \

logdir=logdir/$seqname-$logname2
CUDA_VISIBLE_DEVICES=$dev python lab4d/render_intermediate.py --testdir $logdir --data_class fg
CUDA_VISIBLE_DEVICES=$dev python projects/ppr/export.py --flagfile=$logdir/opts.log --load_suffix latest --inst_id 0 --vis_thresh -20
CUDA_VISIBLE_DEVICES=$dev python lab4d/render_mesh.py --testdir $logdir/export_0000/ --view bev --ghosting
CUDA_VISIBLE_DEVICES=$dev python lab4d/render.py --flagfile=$logdir/opts.log --load_suffix latest --viewpoint rot-0-360 --render_res 256 --freeze_id 0 --num_frames 10