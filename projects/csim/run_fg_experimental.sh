seqname=$1
dev=$2
fg_motion=bob

logname=$fg_motion-l2
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname \
  --fg_motion $fg_motion --num_rounds 20 --feature_type cse --intrinsics_type const --imgs_per_gpu 128 --init_scale_fg 0.1 --extrinsics_type mlp_nodelta --reg_l2_motion_wt 0.02

logdir=logdir/$seqname-$logname
CUDA_VISIBLE_DEVICES=$dev python lab4d/render_intermediate.py --testdir $logdir --data_class fg
CUDA_VISIBLE_DEVICES=$dev python projects/ppr/export.py --flagfile=$logdir/opts.log --load_suffix latest --inst_id 0 --vis_thresh -20
CUDA_VISIBLE_DEVICES=$dev python lab4d/render_mesh.py --testdir $logdir/export_0000/ --view bev --ghosting
CUDA_VISIBLE_DEVICES=$dev python lab4d/render.py --flagfile=$logdir/opts.log --load_suffix latest --viewpoint rot-0-360 --render_res 256 --freeze_id 0 --num_frames 10