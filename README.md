# DepthAI V2VPoseNet
This repo's aim is to get V2VPoseNet working on the DepthAI cameras

## Requirements
Tested on a Windows 11 AMD 5950x Nvidia 3090 machine running:
* Python 3.9.9
* numpy 1.22.0
* open3d 0.14.1.0
* depthai 2.13.3.0
* torch 1.10+cu113 (pytorch)

## How to install
1. Clone the repo
2. Open a Python terminal in the root directory of the repo
3. Run the following to install the dependencies  
   ```python3 install_requirements.py```
4. Install PyTorch (with CUDA) the install link for this on Windows 10/11 with a modern Nvidia GPU is as follows:  
   ```pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html```  
   For other installs use the config tool at [pytorch.org](https://pytorch.org/) to download it

## How to run
Run the following:  
`python3 main.py`  
This should spawn a window of the depth camera output, where the person is highlighted in blue and the background in red.   
Using the blue pointcloud a center point is generated which is passed to V2VPoseNet along with a cropped pointcloud.


## Original V2VPoseNet authors
Moon, Gyeongsik, Ju Yong Chang, and Kyoung Mu Lee. **"V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map."** <i>CVPR 2018. </i> [[arXiv](https://arxiv.org/abs/1711.07399)]
  
  ```
@InProceedings{Moon_2018_CVPR_V2V-PoseNet,
  author = {Moon, Gyeongsik and Chang, Juyong and Lee, Kyoung Mu},
  title = {V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```
