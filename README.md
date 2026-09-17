# STUDENT-CREATIVITY-PROGRAM-PKM_KC-CODA-AI-
SISTEM ASISTEN PENGAWASAN BAYI BERBASIS DEEP LEARNING MULTIMODAL AUDIOVISUAL SEBAGAI SOLUSI PENGAWASAN BAGI ORANGTUA TUNARUNGU
PLEASE READ THIS TEXT TO RUN CODA AI 
Reference : 

Make Sure
install python 3.10

make virtual environment this project
> python -m venv codaai_env

to activate virtual environment 
> source ~/location/env/codaai_env/bin/activate


To train with YOLO please install ultralytic
> sudo apt update
> pip install ultralytics 

To labelling / annotation image to 
> pip install labelme

in this program using oriented bounding box (OBB) to annotation

after annotation you can do augmentation dataset or split dataset with

80% train 20 validation
```
data_directory/
├── images/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
├── labels/
│   ├── class1/
│   │   ├── image1.txt
│   │   └── image2.txt
│   └── class2/
│       ├── image1.txt
│       └── image2.txt
└── classes.json
```
to inference (run) model using Hailo 8L you must quantize into format .hef (Hailo Executable Format)


to quantizazing to hef you must install
install Hailo AI Software Suite – Docker   (at your pc for quantizing your .py model to hef)
install HailoRT – PCIe driver Ubuntu package (deb) (at your raspberry pi to drive hailo 8l board module)


https://hailo.ai/developer-zone/documentation/hailo-sw-suite-2025-10-for-hailo-8-8l/?sp_referrer=suite/suite_install.html#docker-installation


using hailo ai suite docker



docker cp hailo_gpu_v2:/local/workspace/yolov8n.hef ~/codaai/models/

scp ~/codaai/models/yolov8n_mymodel18_lvl4.hef codaai@codaai.local:~/coda_ai/models/

# RasPi_YOLO

Training and deploying YOLO object detection models on Raspberry Pi AI Camera or Hailo AI HAT devices.

# Prerequisites
## For Hailo accelerator (AI HAT)
### On your PC
You'll need to create an account and install the Hailo Dataflow Compiler and HailoRT (python whl AND .deb if using Ubuntu) <br>
https://hailo.ai/developer-zone/software-downloads/ <br>
git clone and run setup script for hailo model zoo (i've had issues with installing from hailo.ai) <br>
https://github.com/hailo-ai/hailo_model_zoo/<br>
Though I highly recommend using their docker container instead <br>
https://hailo.ai/developer-zone/documentation/hailo-sw-suite-2025-01/?sp_referrer=suite/suite_install.html#docker-installation
### On the RasPi 5
Install raspi Hailo software
```
sudo apt install hailo-all
```

See guide for more info <br>
https://www.raspberrypi.com/documentation/computers/ai.html
<br>
## For the AI Camera (Sony IMX500)
### On your PC
If you are using a <b>YOLO8 model</b> Ultralytics onnx exporter can also convert the model to the correct format for the IMX500. <br>
https://docs.ultralytics.com/integrations/sony-imx500/ <br>
<b>If you are using some other model</b> you'll need to install Sony’s Model Compression Toolkit.
```
pip install model_compression_toolkit
```
And use the Pytorch converter tool
```
pip install imx500-converter[pt]
```

And convert/export your model as seen here <br>
https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/mct_features_notebooks/pytorch/example_pytorch_post_training_quantization.ipynb <br>
I have not yet tested this for other model types (aka YOLO11) #TODO
### On the RasPi
Install the RasPi IMX tools (you only need to run imx500-all)
```
sudo apt install imx500-all
```

## Demos
Raspberry Pi's picamera2 has the functionality and demo Python scripts that you are use once you have done the final conversions <br>
Clone picamera2 repository, install it:<br>

```
git clone -b next https://github.com/raspberrypi/picamera2
cd picamera2
pip install -e .  --break-system-packages
```

# Training/Exporting 
## Dataset Preparation

Organize your images in the following structure:
```



Create train/test/validation splits:
```
python create_data_splits.py --data_dir /path/to/data_directory
```

For ONNX export configuration:
```
python create_data_splits.py --data_dir /path/to/data_directory --onnx_config
```

## Training

Train a new model:
```
python yolo_train.py \
    --config /path/to/dataset_data.yaml --name my_model --epoch 100 --batch -1 --image_size 736x736
```
```

Resume training:
```
python yolo_train.py \
    --config /path/to/dataset_config.yaml \
    --name my_model \
    --resume_training
```

## Model ONNX Export

Export to ONNX format:
```
python yolo_train.py \
    --init_model /path/to/trained_model.pt \
    --export_format onnx \
    --export_config /path/to/onnx_config.yaml \
    --export_only
```

Export for IMX500 quantization:
```
python yolo_train.py \
    --init_model /path/to/trained_model.pt \
    --export_format imx \
    --export_config /path/to/onnx_config.yaml \
    --export_only \
    --int8_weights
```

## Calibration Data Generation

Generate calibration data for Hailo converter:
```
python hailo_calibration_data.py \
    --data_dir /path/to/data_directory \
    --target_dir calib \
    --image_size 640, 640 \
    --num_images 1024
```

# Convert the models
## AI Camera (IMX500)
Copy the `packerOut.zip` file and `labels.txt` file from your PC <b>to the RasPi</b> and convert it to the RPK file format <br>
```
imx500-package -i <path to packerOut.zip> -o <output folder>
```
For testing you can connect the IMX500 to the Raspberry Pi and use picamera2 IMX500 demo script!

```
cd picamera2/examples/imx500/
python imx500_object_detection_demo.py --model <path to network.rpk> --labels <path to labels.txt> --fps 25 --bbox-normalization --ignore-dash-labels --bbox-order xy
```
 
## AI HAT (HAILO)
### On your PC
You'll need to perform the conversion, make sure you have run `hailo_calibration_data.py` to create the calib data (`--calib-path`). <br>
Once the YOLO model has finished training (in previous step) the best model will be exported to the ONNX format with default params, 
Use this file to compile `--ckpt`. 
Point `--yaml` to the network config yaml in the hail_model_zoo repo, aka `hailo_model_zoo/hailo_model_zoo/cfg/networks/yolov8n.yaml`.  <br>
Set the number of classes for your model (`--classes`). <br>
Set the `--hw-arch` to your Hailo accelerator type, Hailo 8 (26 TOPS): `hailo8` or Hailo 8L (13 TOPS) `hailo8l`
```
hailomz compile --ckpt <yolo.onnx> --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/yolov8n.yaml --classes <number of classes> --hw-arch hailo8
```

### On the RasPi 5
Copy the `yolo.hef` file and `labels.txt` file from your PC <b>to the RasPi</b>, connect a camera, and run the Hailo picamera2 demo script!

cd picamera2/examples/hailo/
python detect.py --model <path to yolo.hef> --labels <path to labels.txt>



cat /local/workspace/hailo_model_zoo/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/yolov8n.alls > /local/workspace/hailo_model_zoo/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/boneka6_lvl2.alls


///////////////////
cat > /local/workspace/hailo_model_zoo/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/boneka6_lvl4.alls << 'EOF'
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
model_optimization_config(compression_params, auto_16bit_weights_ratio=0)
model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=8)
nms_postprocess("../../postprocess_config/yolov8n_nms_config.json", meta_arch=yolov8, engine=cpu)
allocator_param(width_splitter_defuse=disabled)
EOF


cat > /local/workspace/hailo_model_zoo/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/27_lvl2.alls << 'EOF'
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
nms_postprocess("../../postprocess_config/yolov8n_nms_config.json", meta_arch=yolov8, engine=cpu)
allocator_param(width_splitter_defuse=disabled)
model_optimization_config(calibration, batch_size=8, calibset_size=256)
model_optimization_flavor(optimization_level=2, compression_level=1)
model_optimization_config(checker_cfg, dataset_size=256)
post_quantization_optimization(finetune, policy=enabled, batch_size=4, dataset_size=256)
EOF

cat /local/workspace/hailo_model_zoo/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/27_lvl2.alls


rm -f /local/workspace/yolov8n.har /local/workspace/*.hef

 hailomz compile yolov8n \
    --ckpt /local/workspace/codaai/RasPi_YOLO-main/runs/detect/boneka-6/weights/best.onnx \
    --calib-path /local/workspace/codaai/train20/calib \
    --classes 8 \
    --hw-arch hailo8l \
    --model-script /local/workspace/hailo_model_zoo/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/mymodel18_lvl2.alls \
    --end-node-names "/model/model.22/cv2.0/cv2.0.2/Conv" "/model/model.22/cv3.0/cv3.0.2/Conv" "/model/model.22/cv2.1/cv2.1.2/Conv" "/model/model.22/cv3.1/cv3.1.2/Conv" "/model/model.22/cv2.2/cv2.2.2/Conv" "/model/model.22/cv3.2/cv3.2.2/Conv"



docker cp hailo_gpu_v2:/local/workspace/yolov8n.har /home/chel/codaai/
docker cp hailo_gpu_v2:/local/workspace/yolov8n.hef /home/chel/codaai/

scp ~/codaai/models/yolov8n_mymodel18_lvl4.hef codaai@codaai.local:~/coda_ai/models/



//////////LEVELS///////////////

cat > /local/workspace/hailo_model_zoo/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/27_lvl2.alls << 'EOF'
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
nms_postprocess("../../postprocess_config/yolov8n_nms_config.json", meta_arch=yolov8, engine=cpu)
allocator_param(width_splitter_defuse=disabled)
EOF


////1////
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
nms_postprocess("../../postprocess_config/yolov8n_nms_config.json", meta_arch=yolov8, engine=cpu)
allocator_param(width_splitter_defuse=disabled)

model_optimization_config(calibration, batch_size=8, calibset_size=512)
model_optimization_flavor(optimization_level=1, compression_level=1)

////2////
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
nms_postprocess("../../postprocess_config/yolov8n_nms_config.json", meta_arch=yolov8, engine=cpu)
allocator_param(width_splitter_defuse=disabled)

model_optimization_config(calibration, batch_size=8, calibset_size=1024)
model_optimization_flavor(optimization_level=2, compression_level=1)
model_optimization_config(checker_cfg, dataset_size=1024)
model_optimization_config(checker_cfg, batch_size=1)
post_quantization_optimization(finetune, policy=enabled, batch_size=4, dataset_size=1024)



//////3/////
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
nms_postprocess("../../postprocess_config/yolov8n_nms_config.json", meta_arch=yolov8, engine=cpu)
allocator_param(width_splitter_defuse=disabled)

model_optimization_config(calibration, batch_size=8, calibset_size=1024)
post_quantization_optimization(adaround, policy=enabled, dataset_size=1024)
model_optimization_flavor(optimization_level=3, compression_level=1)
model_optimization_config(checker_cfg, dataset_size=1024)
model_optimization_config(checker_cfg, batch_size=1)
post_quantization_optimization(finetune, policy=enabled, batch_size=1, dataset_size=1024)




/////4/////
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
nms_postprocess("../../postprocess_config/yolov8n_nms_config.json", meta_arch=yolov8, engine=cpu)
allocator_param(width_splitter_defuse=disabled)

model_optimization_config(calibration, batch_size=8, calibset_size=1024)
post_quantization_optimization(adaround, policy=enabled, dataset_size=1024)
model_optimization_flavor(optimization_level=4, compression_level=1)
model_optimization_config(checker_cfg, dataset_size=1024)
model_optimization_config(checker_cfg, batch_size=1)
post_quantization_optimization(finetune, policy=enabled, batch_size=1, dataset_size=1024)

===========================================================================================================
cat > /local/workspace/hailo_model_zoo/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/28_lvl4.alls << 'EOF'
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
nms_postprocess("../../postprocess_config/yolov8n_nms_config.json", meta_arch=yolov8, engine=cpu)
allocator_param(width_splitter_defuse=disabled)

model_optimization_config(calibration, batch_size=2, calibset_size=1260)
post_quantization_optimization(adaround, policy=enabled, dataset_size=1260)
model_optimization_flavor(optimization_level=4, compression_level=1)
model_optimization_config(checker_cfg, dataset_size=1260)
model_optimization_config(checker_cfg, batch_size=1)
post_quantization_optimization(finetune, policy=enabled, batch_size=1, dataset_size=1260)
EOF


