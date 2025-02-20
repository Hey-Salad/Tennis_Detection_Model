# Tennis_Detection_Model
This repository contains a YOLOv8-based tennis detection model that can detect the tennis ball, the player, and the tennis racket and it has been converted for use with SeeedStudio reCamera.

## Environment Setup

### Hardware Requirements
The conversion process was performed on a system with NVIDIA GPU support. Here's the GPU configuration used:

```bash
$ nvidia-smi
Tue Feb 11 22:23:48 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla P100-PCIE-16GB           On  |   00000000:00:04.0 Off |                    0 |
| N/A   32C    P0             27W /  250W |       3MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### Software Requirements

- Ubuntu 22.04
- Python 3.10
- Docker
- tpu-mlir v1.15.1-20250208

## Step-by-Step Conversion Guide

### 1. Setup Docker Environment

Pull the required Docker image:
```bash
$ docker pull sophgo/tpuc_dev:v3.1

v3.1: Pulling from sophgo/tpuc_dev
b237fe92c417: Pull complete 
db3c30810eab: Pull complete 
2651dfd68288: Pull complete 
[...]
Digest: sha256:9f3b2244d09ee3ec4b9e039484a9a3c1e419edcc8f64b86dd61da6a87565741c
Status: Downloaded newer image for sophgo/tpuc_dev:v3.1
docker.io/sophgo/tpuc_dev:v3.1
```

Create and enter the Docker container:
```bash
$ docker run --privileged --name tennis_model -v $PWD:/workspace -it sophgo/tpuc_dev:v3.1
```

### 2. Install Required Dependencies

Inside the Docker container, install tpu_mlir:
```bash
$ pip install tpu_mlir[all]

Collecting tpu_mlir[all]
  Downloading tpu_mlir-1.15.1-py3-none-any.whl (216.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 216.6/216.6 MB 14.7 MB/s eta 0:00:00
Successfully installed tpu_mlir-1.15.1
```

### 3. Prepare the Environment

Clone and set up tpu-mlir:
```bash
$ cd /workspace
$ git clone https://github.com/sophgo/tpu-mlir.git
Cloning into 'tpu-mlir'...
remote: Enumerating objects: 98293, done.
remote: Counting objects: 100% (115/115), done.
remote: Compressing objects: 100% (104/104), done.
remote: Total 98293 (delta 51), reused 13 (delta 11), pack-reused 98178 (from 3)
Receiving objects: 100% (98293/98293), 3.31 GiB | 35.54 MiB/s, done.
Resolving deltas: 100% (65627/65627), done.
```

Set up the environment:
```bash
$ cd tpu-mlir
$ source ./envsetup.sh
PROJECT_ROOT : /workspace/tpu-mlir
BUILD_PATH   : /workspace/tpu-mlir/build
INSTALL_PATH : /workspace/tpu-mlir/install
[...]
$ ./build.sh
```

### 4. Model Conversion Process

#### Create Working Directory and Copy Model
```bash
$ mkdir model_tennis && cd model_tennis
$ cp ../Tennis_ball_detection_model/best.onnx .
$ mkdir workspace && cd workspace
```

#### Convert ONNX to MLIR
```bash
$ model_transform \
    --model_name tennis_detect \
    --model_def ../best.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean "0.0,0.0,0.0" \
    --scale "0.0039216,0.0039216,0.0039216" \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --mlir tennis_detect.mlir

2025/02/12 06:54:40 - INFO : TPU-MLIR v1.15.1-20250208
2025/02/12 06:54:40 - INFO : 
         _____________________________________________________ 
        | preprocess:                                           |
        |   (x - mean) * scale                                  |
        '-------------------------------------------------------'
  config Preprocess args : 
        resize_dims           : same to net input dims
        keep_aspect_ratio     : True
        keep_ratio_mode       : letterbox
        pad_value             : 0
        pad_type             : center
        --------------------------
        mean                  : [0.0, 0.0, 0.0]
        scale                 : [0.0039216, 0.0039216, 0.0039216]
        --------------------------
        pixel_format          : rgb
        channel_format        : nchw

[... Model conversion progress messages ...]
2025/02/12 06:54:41 - INFO : Mlir file generated:tennis_detect.mlir
```

#### Prepare Calibration Data
```bash
$ mkdir -p COCO2017/images
$ wget http://images.cocodataset.org/val2017/000000000139.jpg -P COCO2017/images/
--2025-02-12 06:59:35--  http://images.cocodataset.org/val2017/000000000139.jpg
Resolving images.cocodataset.org... 52.217.199.81
Connecting to images.cocodataset.org|52.217.199.81|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 161811 (158K) [image/jpeg]
Saving to: 'COCO2017/images/000000000139.jpg'

000000000139.jpg     100%[======================>] 158.02K  --.-KB/s    in 0.08s   

2025-02-12 06:59:35 (1.83 MB/s) - 'COCO2017/images/000000000139.jpg' saved [161811/161811]

[... Repeat for other calibration images ...]
```

#### Run Calibration
```bash
$ run_calibration \
    tennis_detect.mlir \
    --dataset ../COCO2017/images \
    --input_num 4 \
    -o tennis_detect_calib_table

TPU-MLIR v1.15.1-20250208
input_num = 4, ref = 4
real input_num = 4
activation_collect_and_calc_th for sample: 0:   0%|           | 0/4 [00:00<?, ?it/s]
[##################################################] 100%
activation_collect_and_calc_th for sample: 1:  50%|█▌ | 2/4 [00:01<00:01,  1.74it/s]
[##################################################] 100%
[... Calibration progress ...]
```

#### Convert to INT8 Model
```bash
$ model_deploy \
    --mlir tennis_detect.mlir \
    --quantize INT8 \
    --chip cv181x \
    --calibration_table tennis_detect_calib_table \
    --model tennis_detect_int8.cvimodel

2025/02/12 07:03:15 - INFO : TPU-MLIR v1.15.1-20250208
[... Model deployment progress ...]
```

### 5. Verify Converted Model

Check the model information:
```bash
$ model_tool --info tennis_detect_int8.cvimodel

Mlir Version: v1.15.1-20250208
Cvimodel Version: 1.4.0
tennis_detect Build at 2025-02-12 07:03:16
For cv181x chip ONLY
CviModel Need ION Memory Size: (11.73 MB)

Sections:
ID   TYPE      NAME                     SIZE        OFFSET      MD5
000  weight    weight                   3120192     0           b9f35851c394615733555613578a6b3e
001  cmdbuf    subfunc_1                476232      3120192     78bc8b80b07e2b512a50f5c601ef8dbe
[...]
```

## Model Specifications

### Input Requirements
- Shape: [1, 3, 640, 640]
- Format: RGB
- Preprocessing:
  - Mean: [0.0, 0.0, 0.0]
  - Scale: [0.0039216, 0.0039216, 0.0039216]
  - Keep aspect ratio: True
  - Pad type: center

### Memory Requirements
- ION Memory Size: 11.73 MB
- Private GMEM Size: 1228800 bytes
- Shared GMEM Size: 2395600 bytes

## Common Issues and Solutions

### Warning Messages
During conversion, you might see these warnings:
```
WARNING : ConstantFolding failed.
WARNING : onnxsim opt failed.
```
These are normal and don't affect the final model.

### Calibration Issues
If calibration fails, ensure:
- Calibration images are accessible
- Images are valid JPEG/PNG files
- At least 4 images are available

### Docker Volume Mounting
If you can't see files in Docker, check:
```bash
$ docker run --privileged --name tennis_model -v ${PWD}:/workspace -it sophgo/tpuc_dev:v3.1
```
The -v flag mounts your current directory to /workspace.

## File Structure After Conversion
```
.
├── best.onnx                    # Original model
├── tennis_detect.mlir           # Intermediate MLIR representation
├── tennis_detect_calib_table    # Calibration data
├── tennis_detect_int8.cvimodel  # Final converted model
└── COCO2017/
    └── images/                  # Calibration images
```

## License

[Add your license information here]

## Contributing

Feel free to open issues or submit pull requests for improvements.

## Acknowledgments

- YOLO team for the original architecture
- Sophgo team for the tpu-mlir toolchain
- COCO dataset for calibration images
- Special Thanks to the Seeedstudio reCamera team Ddddawn and Leel
- Special Thanks to Heysalad Team especially Chenran our AI intern who began the project
