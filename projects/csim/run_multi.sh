# envname=Oct5at10-49AM-poly
# envname=Feb26at10-02 PM-poly
envname=Feb14at5-55PM-poly
#envname=Feb19at9-47 PM-poly
fg_motion=urdf-quad
seqname=$1
dev=$2

# bash projects/csim/recon_scene.sh $envname $dev
# bash projects/csim/align_scene_multi.sh $envname $seqname $dev
bash projects/csim/recon_compose_multi.sh $seqname $fg_motion $dev
