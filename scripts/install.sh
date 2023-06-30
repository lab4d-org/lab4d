if ! command -v micromamba &> /dev/null
then
    curl micro.mamba.pm/install.sh | bash
fi

micromamba create -f environment.yml --yes

eval "$(micromamba shell hook --shell=bash)" && micromamba activate lab4d

mim install mmcv

(cd preprocess/third_party/MinVIS/mask2former/modeling/pixel_decoder/ops && sh make.sh)

(cd lab4d/third_party/quaternion && pip install .)

mkdir ./preprocess/third_party/Track-Anything/checkpoints; gdown --fuzzy https://drive.google.com/uc?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3 -O ./preprocess/third_party/Track-Anything/checkpoints/E2FGVI-HQ-CVPR22.pth

wget https://www.dropbox.com/s/bgsodsnnbxdoza3/vcn_rob.pth -O ./preprocess/third_party/vcnplus/vcn_rob.pth

gdown --fuzzy https://drive.google.com/file/d/1j46M_NFGzpt2Ga4ptOumTRAjmr8eQb1Y/view?usp=sharing -O ./preprocess/third_party/MinVIS/demo_video/minvis_ytvis21_swin_large.pth

wget https://www.dropbox.com/s/51cjzo8zgz966t5/human.pth -O preprocess/third_party/viewpoint/human.pth

wget https://www.dropbox.com/s/1464pg6c9ce8rve/quad.pth -O preprocess/third_party/viewpoint/quad.pth

