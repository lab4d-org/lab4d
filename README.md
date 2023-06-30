<p>
  <picture>
  <img alt="logo" src="https://github.com/gengshan-y/lab4d/assets/13134872/5f7d24dd-fd28-4459-bd58-d51f9bb79cf1" width="350px" />
  </picture>
</p>

# Lab4D
**[[Docs & Tutorials](https://lab4d-org.github.io/lab4d/)]**

*This is an alpha release and the APIs are subject to change. Please provide feedback and report bugs via github issues. Thank you for your support.*

## TODOs
- [ ] more data and checkpoints
- [ ] feedforward models
- [ ] web viewer
- [ ] evaluation and benchmarks
- [ ] multi-view reconstruction

## About
Lab4D is a pipeline for dynamic 3D reconstruction (aka 4D capture) from monocular videos. The software is licensed under the MIT license. 

If you use this project for your research, please consider citing the following papers. For building deformable object models, cite:
```
@inproceedings{yang2022banmo,
  title={BANMo: Building Animatable 3D Neural Models from Many Casual Videos},
  author={Yang, Gengshan and Vo, Minh and Neverova, Natalia and Ramanan, Deva and Vedaldi, Andrea and Joo, Hanbyul},
  booktitle = {CVPR},
  year={2022}
}  
```

For building category body and pose models, cite:
```
@inproceedings{yang2023rac,
    title={Reconstructing Animatable Categories from Videos},
    author={Yang, Gengshan and Wang, Chaoyang and Reddy, N. Dinesh and Ramanan, Deva},
    booktitle = {CVPR},
    year={2023}
} 
```

For object-scene reconstruction and extreme view synthesis, cite:
```
@article{song2023totalrecon,
  title={Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis},
  author={Song, Chonghyuk and Yang, Gengshan and Deng, Kangle and Zhu, Jun-Yan and Ramanan, Deva},
  journal={arXiv},
  year={2023}
}
```

For training feed-forward video/image shape and pose estimators, cite:
```
@inproceedings{tan2023distilling,
  title={Distilling Neural Fields for Real-Time Articulated Shape Reconstruction},
  author={Tan, Jeff and Yang, Gengshan and Ramanan, Deva},
  booktitle={CVPR},
  year={2023}
}
```


## Acknowledgement
- We thank [@mjlbach](https://github.com/mjlbach), [@alexanderbergman7](https://github.com/alexanderbergman7), and [@terrancewang](https://github.com/terrancewang) for testing and feedback
- We thank [@jasonyzhang](https://github.com/jasonyzhang), [@MightyChaos](https://github.com/MightyChaos), [@JudyYe](https://github.com/JudyYe), and [@andrewsonga](https://github.com/andrewsonga) for feedback
- Our pre-processing pipeline is built upon the following open-sourced repos: 
  - Segmentation: [Track-Anything](https://github.com/gaomingqi/Track-Anything), [MinVIS](https://github.com/NVlabs/MinVIS)
  - Feature & correspondence: [DensePose-CSE](https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/projects/DensePose/doc/DENSEPOSE_CSE.md), [DINOv2](https://github.com/facebookresearch/dinov2), [VCNPlus](https://github.com/gengshan-y/rigidmask)
  - Depth: [ZoeDepth](https://github.com/isl-org/ZoeDepth)
  - Camera: [BANMo-viewpoint](https://github.com/facebookresearch/banmo)
- We use [dqtorch](https://github.com/MightyChaos/dqtorch) for efficient rotation operations
