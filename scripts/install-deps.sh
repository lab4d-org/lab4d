(cd lab4d/third_party/quaternion && pip install .)

mkdir ./preprocess/third_party/Track-Anything/checkpoints; gdown --fuzzy https://drive.google.com/uc?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3 -O ./preprocess/third_party/Track-Anything/checkpoints/E2FGVI-HQ-CVPR22.pth

wget https://www.dropbox.com/s/bgsodsnnbxdoza3/vcn_rob.pth -O ./preprocess/third_party/vcnplus/vcn_rob.pth

wget https://www.dropbox.com/s/51cjzo8zgz966t5/human.pth -O preprocess/third_party/viewpoint/human.pth

wget https://www.dropbox.com/s/1464pg6c9ce8rve/quad.pth -O preprocess/third_party/viewpoint/quad.pth