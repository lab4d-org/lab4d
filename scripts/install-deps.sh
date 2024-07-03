mim install mmcv

(cd lab4d/third_party/quaternion && CUDA_HOME=$CONDA_PREFIX pip install .)

mkdir ./preprocess/third_party/Track-Anything/checkpoints; wget "https://www.dropbox.com/scl/fi/o86gx6zn27b494m937n2i/E2FGVI-HQ-CVPR22.pth?rlkey=j15ue65ryy8jb1mvn2htf0jtk&st=t4zyl5jk&dl=0" -O ./preprocess/third_party/Track-Anything/checkpoints/E2FGVI-HQ-CVPR22.pth

wget https://www.dropbox.com/s/bgsodsnnbxdoza3/vcn_rob.pth -O ./preprocess/third_party/vcnplus/vcn_rob.pth

wget https://www.dropbox.com/s/51cjzo8zgz966t5/human.pth -O preprocess/third_party/viewpoint/human.pth

wget https://www.dropbox.com/s/1464pg6c9ce8rve/quad.pth -O preprocess/third_party/viewpoint/quad.pth
