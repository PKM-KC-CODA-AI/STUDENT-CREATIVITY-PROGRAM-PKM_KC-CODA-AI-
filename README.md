# STUDENT-CREATIVITY-PROGRAM-PKM_KC-CODA-AI-
SISTEM ASISTEN PENGAWASAN BAYI BERBASIS DEEP LEARNING MULTIMODAL AUDIOVISUAL SEBAGAI SOLUSI PENGAWASAN BAGI ORANGTUA TUNARUNGU
PLEASE READ THIS TEXT TO RUN CODA AI 
Reference : 
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

to inference (run) model using Hailo 8L you must quantize into format .hef (Hailo Executable Format)


to quantizazing to hef you must install

