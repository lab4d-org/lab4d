envname=Oct5at10-49AM-poly
seqname=$1
dev=$2

# python projects/csim/rerun_dinov2.py $seqname $dev
# bash projects/csim/recon_scene.sh $envname $dev
bash projects/csim/align_scene_multi.sh $envname $seqname $dev
bash projects/csim/recon_compose_multi_smpl.sh $seqname $dev # human
