seqname=$1
fg_motion=$2
dev=$3

# ft2: 10x depth loss, also n_depth=256
logname=compose-ft2-totalrecon
rm -rf logdir/$seqname-$logname
bash scripts/train.sh lab4d/train.py $dev --seqname $seqname --logname $logname --field_type comp --fg_motion $fg_motion \
  --intrinsics_type const --extrinsics_type mixse3 \
  --freeze_scale --freeze_camera_bg --freeze_field_fgbg --learning_rate 1e-4 --noreset_steps --noabsorb_base --nouse_freq_anneal --num_rounds 20 \
  --load_path_bg logdir/$seqname-bg-totalrecon-v2/ckpt_latest.pth --load_path logdir/$seqname-compose-ft/ckpt_latest.pth \
  --mask_wt 0.1 --normal_wt 0.0 --depth_wt 1e-2 --reg_eikonal_wt 1e-3 \
  --pixels_per_image 10 --bg_vid 0 \
  --scene_type share-x --beta_prob_init_bg 0.0 --beta_prob_final_bg 0.0 --beta_prob_init_fg 1.0 --beta_prob_final_fg 1.0  --nomlp_init \
  --imgs_per_gpu 128 --feature_type cse

  # --load_path_bg logdir/$seqname-bg-adapt4/ckpt_latest.pth --load_path logdir/$seqname-compose-ft/ckpt_latest.pth \
