# Description: Run experiments for PPR
# skeleton
# rm -rf logdir/cat-pikachu-0-ppr-comp-skel
# bash scripts/train.sh lab4d/train.py 2 --seqname cat-pikachu-0 --logname ppr-comp-skel --field_type comp --fg_motion skel-quad --data_prefix full --num_rounds 120

# cat-pikachu-0
# rm -rf logdir/cat-pikachu-0-bg-r60-eik-n3-e2
# bash scripts/train.sh lab4d/train.py 0 --seqname cat-pikachu-0 --logname bg-r60-eik-n3-e2 \
#  --field_type bg --data_prefix full --num_rounds 60 --alter_flow --mask_wt 0.01 --normal_wt 1e-3 --reg_eikonal_wt 0.01

# rm -rf logdir/cat-pikachu-0-fg-urdf-eik-g1-sinc-couple
# bash scripts/train.sh lab4d/train.py 1 --seqname cat-pikachu-0 --logname fg-urdf-eik-g1-sinc-couple --fg_motion urdf-quad --num_rounds 20 --feature_type cse
# rm -rf logdir/cat-pikachu-0-fg-urdf-ft
# bash scripts/train.sh lab4d/train.py 0 --seqname cat-pikachu-0 --logname fg-urdf-ft --fg_motion urdf-quad --num_rounds 20 --feature_type cse --depth_wt 0.0 --reg_eikonal_wt 0.0 \
#     --load_path logdir/cat-pikachu-0-fg-urdf-eik-g1/ckpt_latest.pth --pose_correction --noreset_steps

# rm -rf logdir/cat-pikachu-0-ppr-exp2
# bash scripts/train.sh projects/ppr/train.py 0 --seqname cat-pikachu-0 --logname ppr-exp2 --field_type comp --fg_motion urdf-quad --feature_type cse \
#     --num_rounds 20  --iters_per_round 100 --ratio_phys_cycle 0.5 --phys_vis_interval 20 --secs_per_wdw 2.4 \
#     --pixels_per_image 12 --noreset_steps --learning_rate 1e-4 --noabsorb_base \
#     --load_path logdir/cat-pikachu-0-fg-urdf-eik-g1-sinc/ckpt_latest.pth \
#     --load_path_bg logdir/cat-pikachu-0-bg-r60-eik-n3-e2/ckpt_latest.pth


# ama-samba
# python scripts/run_preprocess.py ama-samba-4v human "0,2";

# rm -rf logdir/ama-samba-4v-bg
# bash scripts/train.sh lab4d/train.py 0 --seqname ama-samba-4v --logname bg \
#   --field_type bg --data_prefix full --num_rounds 20 --alter_flow --mask_wt 0.01 --normal_wt 1e-3 --reg_eikonal_wt 0.01 --scene_type sep-x --freeze_intrinsics 

# rm -rf logdir/ama-samba-4v-fg-urdf-eik2
# bash scripts/train.sh lab4d/train.py 0 --seqname ama-samba-4v --logname fg-urdf-eik2 --fg_motion urdf-human --num_rounds 60 --feature_type cse --freeze_intrinsics
# rm -rf logdir/ama-samba-4v-fg-urdf-r60-fixproj-ft
# bash scripts/train.sh lab4d/train.py 0 --seqname ama-samba-4v --logname fg-urdf-r60-fixproj-ft --fg_motion urdf-human --num_rounds 20 --feature_type cse --depth_wt 0.0 \
#     --load_path logdir/ama-samba-4v-fg-urdf-r60-fixproj/ckpt_latest.pth --pose_correction --noreset_steps

# rm -rf logdir/ama-samba-4v-ppr-exp2
# bash scripts/train.sh projects/ppr/train.py 1 --seqname ama-samba-4v --logname ppr-exp2 --field_type comp --fg_motion urdf-human --feature_type cse --scene_type sep-x \
#     --num_rounds 20  --iters_per_round 100 --ratio_phys_cycle 0.5 --phys_vis_interval 20 --frame_interval 0.0333 --secs_per_wdw 2.0 \
#     --pixels_per_image 12 --noreset_steps --learning_rate 1e-4 --noabsorb_base \
#     --load_path logdir/ama-samba-4v-fg-urdf-eik2/ckpt_latest.pth \
#     --load_path_bg logdir/ama-samba-4v-bg/ckpt_latest.pth

# *****ama-bouncing
# python scripts/run_preprocess.py ama-bouncing-4v human "0,2";

# rm -rf logdir/ama-bouncing-4v-bg
# bash scripts/train.sh lab4d/train.py 2 --seqname ama-bouncing-4v --logname bg \
#   --field_type bg --data_prefix full --num_rounds 20 --alter_flow --mask_wt 0.01 --normal_wt 1e-3 --reg_eikonal_wt 0.01 --scene_type sep-x --freeze_intrinsics

# rm -rf logdir/ama-bouncing-4v-fg-urdf-eik
# bash scripts/train.sh lab4d/train.py 2 --seqname ama-bouncing-4v --logname fg-urdf-eik --fg_motion urdf-human --num_rounds 60 --feature_type cse --freeze_intrinsics
# rm -rf logdir/ama-bouncing-4v-fg-urdf-r60-fixproj-ft
# bash scripts/train.sh lab4d/train.py 0 --seqname ama-bouncing-4v --logname fg-urdf-r60-fixproj-ft --fg_motion urdf-human --num_rounds 20 --feature_type cse --depth_wt 0.0 \
#     --load_path logdir/ama-bouncing-4v-fg-urdf-r60-fixproj/ckpt_latest.pth --pose_correction --noreset_steps

# rm -rf logdir/ama-bouncing-4v-ppr-exp
# bash scripts/train.sh projects/ppr/train.py 1 --seqname ama-bouncing-4v --logname ppr-exp --field_type comp --fg_motion urdf-human --feature_type cse --scene_type sep-x \
#     --num_rounds 20  --iters_per_round 100 --ratio_phys_cycle 0.5 --phys_vis_interval 20 --frame_interval 0.0333 --secs_per_wdw 1.0 \
#     --pixels_per_image 12 --noreset_steps --learning_rate 1e-4 --noabsorb_base \
#     --load_path logdir/ama-bouncing-4v-fg-urdf-eik/ckpt_latest.pth \
#     --load_path_bg logdir/ama-bouncing-4v-bg/ckpt_latest.pth

    # --load_path logdir/ama-bouncing-4v-fg-urdf-r60-intrinsics/ckpt_latest.pth \
    # --load_path logdir/ama-bouncing-4v-fg-urdf-r60-se3-sh-fix-reproj-fixshape/ckpt_latest.pth \
    # --load_path_bg logdir/ama-bouncing-4v-bg-intrinsics/ckpt_latest.pth
    # --load_path_bg logdir/ama-bouncing-4v-bg-normal-r60-clip-aabb/ckpt_latest.pth

# shiba-haru: challenging due to the length
rm -rf logdir/shiba-haru-bg
bash scripts/train.sh lab4d/train.py 1,2 --seqname shiba-haru --logname bg \
    --field_type bg --data_prefix full --num_rounds 60 --alter_flow --mask_wt 0.01 --normal_wt 1e-3 --reg_eikonal_wt 0.01 --scene_type sep-x

rm -rf logdir/shiba-haru-fg-urdf
bash scripts/train.sh lab4d/train.py 1,2 --seqname shiba-haru --logname fg-urdf --fg_motion urdf-quad --num_rounds 60 --feature_type cse
rm -rf logdir/shiba-haru-fg-urdf-proj
bash scripts/train.sh lab4d/train.py 1,2 --seqname shiba-haru --logname fg-urdf-proj --fg_motion urdf-quad --num_rounds 20 --feature_type cse \
    --load_path logdir/shiba-haru-fg-urdf/ckpt_latest.pth --pose_correction --noreset_steps

rm -rf logdir/shiba-haru-ppr
bash scripts/train.sh projects/ppr/train.py 1 --seqname shiba-haru --logname ppr --field_type comp --fg_motion urdf-quad --feature_type cse --scene_type sep-x \
    --num_rounds 20  --iters_per_round 100 --ratio_phys_cycle 0.5 --phys_vis_interval 20 --frame_interval 0.0666 --secs_per_wdw 2.4 \
    --pixels_per_image 12 --noreset_steps --learning_rate 1e-4  --noabsorb_base --phys_vid 9 \
    --load_path logdir/shiba-haru-fg-urdf-proj/ckpt_latest.pth \
    --load_path_bg logdir/shiba-haru-bg/ckpt_latest.pth

# # squirrel: challenging due to the camera initialization
# python scripts/run_preprocess.py squirrel squirrel "0";
# rm -rf logdir/squirrel-fg-urdf-cse
# bash scripts/train.sh lab4d/train.py 2 --seqname squirrel --logname fg-urdf-cse --fg_motion urdf-quad --num_rounds 20 --feature_type cse
# rm -rf logdir/squirrel-bg
# bash scripts/train.sh lab4d/train.py 2 --seqname squirrel --logname bg --field_type bg --data_prefix full --num_rounds 20
# rm -rf logdir/squirrel-ppr-comp-urdf-phys
# bash scripts/train.sh projects/ppr/train.py 2 --seqname squirrel --logname ppr-comp-urdf-phys --field_type comp --fg_motion urdf-quad --feature_type cse \
#     --secs_per_wdw 2.0 --num_rounds 10  --iters_per_round 200 --ratio_phys_cycle 0.5 --nouse_freq_anneal \
#     --load_path logdir/squirrel-fg-urdf-cse/ckpt_latest.pth \
#     --load_path_bg logdir/squirrel-bg/ckpt_latest.pth

# # dog-robolounge: challenging due to the length
# rm -rf logdir/dog-robolounge-fg-urdf-cse
# bash scripts/train.sh lab4d/train.py 2 --seqname dog-robolounge --logname fg-urdf-cse --fg_motion urdf-quad --num_rounds 20 --feature_type cse
# rm -rf logdir/dog-robolounge-bg
# bash scripts/train.sh lab4d/train.py 2 --seqname dog-robolounge --logname bg --field_type bg --data_prefix full --num_rounds 20
# rm -rf logdir/dog-robolounge-ppr-comp-urdf-phys
# bash scripts/train.sh projects/ppr/train.py 1 --seqname dog-robolounge --logname ppr-comp-urdf-phys --field_type comp --fg_motion urdf-quad --feature_type cse \
#     --secs_per_wdw 2.0 --num_rounds 10  --iters_per_round 40 --ratio_phys_cycle 0.5 --nouse_freq_anneal \
#     --load_path logdir/dog-robolounge-fg-urdf-cse/ckpt_latest.pth \
#     --load_path_bg logdir/dog-robolounge-bg/ckpt_latest.pth