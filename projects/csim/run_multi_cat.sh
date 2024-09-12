# envname=Oct5at10-49AM-poly-cp
envname=$1
seqname=$2
dev=$3

python projects/csim/rerun_dinov2.py $seqname $dev
bash projects/csim/recon_scene.sh $envname $dev
bash projects/csim/align_scene_multi.sh $envname $seqname $dev
bash projects/csim/recon_compose_multi.sh $seqname urdf-quad $dev # cat, with shape