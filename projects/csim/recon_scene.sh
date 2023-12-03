# reconstruct a canonical background scene
# input: polycam sequences
envname=$1
dev=$2

# single stage with frozen camera
seqname=$envname
logname=bg
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
  --intrinsics_type const --extrinsics_type const --feature_channels 384 \
  --freeze_scale --learning_rate 2e-3 --num_rounds 240 \
  --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 0.01 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --flow_wt 0.0 \
  --pixels_per_image 4 --imgs_per_gpu 768
  # --init_scale_bg 0.2 
  # --feature_type dinov2-reg
  # --train_res 1024 \
  # --normal_wt 0.0 --feature_wt 0.0 \
  # --reg_eikonal_wt 0.0  --nouse_freq_anneal 

# # 2nd stage to optimize cams
# seqname=$envname
# logname=bg-nonf-1024-2d-Sobol-scramble-long-allf-samp-4-768-ft
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
#   --intrinsics_type const --extrinsics_type explicit --feature_channels 384 \
#   --freeze_scale --learning_rate 1e-4 --num_rounds 240 \
#   --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 0.01 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --flow_wt 0.0 \
#   --num_rounds 240 --normal_wt 0.0 --feature_wt 0.0 --train_res 1024 --reg_eikonal_wt 0.0 \
#   --pixels_per_image 4 --imgs_per_gpu 768 \
#   --load_path logdir/$envname-bg-nonf-1024-2d-Sobol-scramble-long-allf-samp-4-768-2/ckpt_latest.pth --noreset_steps --noabsorb_base \

CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$envname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --grid_size 256



# # initialize field with frozen camera
# seqname=$envname
# logname=bg-init
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
#   --intrinsics_type const --extrinsics_type const --feature_channels 384 \
#   --freeze_scale --learning_rate 1e-3 --num_rounds 20 \
#   --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 0.01 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --flow_wt 2e-3
# CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$envname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --grid_size 256

# # optimize camera as well
# seqname=$envname
# logname=bg-ft
# rm -rf logdir/$seqname-$logname
# bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type bg --data_prefix full \
#   --intrinsics_type const --feature_channels 384 \
#   --freeze_scale --learning_rate 1e-3 --num_rounds 120 --load_path logdir/$envname-bg-init/ckpt_latest.pth --noreset_steps --reset_beta \
#   --mask_wt 0.01 --normal_wt 1e-3 --feature_wt 1e-3 --depth_wt 0.01 --reg_eikonal_wt 0.001 --feat_reproj_wt 0.0 --flow_wt 2e-3
# CUDA_VISIBLE_DEVICE=$dev python lab4d/export.py --flagfile=logdir/$envname-$logname/opts.log --load_suffix latest --inst_id 0 --vis_thresh -10 --grid_size 256