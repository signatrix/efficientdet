# EfficientDet: Scalable and Efficient Object Detection

## Introduction

Here is our pytorch implementation of the model described in the paper **EfficientDet: Scalable and Efficient Object Detection** [paper](https://arxiv.org/abs/1911.09070). 
<p align="center">
  <img src="demo/video.gif"><br/>
  <i>An example of my model's output.</i>
</p>

## How to use my code

With our code, you can:
* **Train your model from scratch**
* **Train your model with my trained model**
* **Evaluate test images/videos with either our trained model or yours**

## Requirements:

* **python 3.6**
* **pytorch 0.4**
* **opencv (cv2)**
* **tensorboard**
* **tensorboardX** (This library could be skipped if you do not use SummaryWriter)
* **pycocotools**
* **efficientnet_pytorch**

## Datasets:


| Dataset                | Classes |    #Train images      |    #Validation images      |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| COCO2017               |    80   |          118k         |              5k            |

Create a data folder under the repository,

```
cd {repo_root}
mkdir data
```
  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:
  ```
  COCO
  ├── annotations
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  │── images
      ├── train2017
      └── val2017
  ```
  
## How to use our code

With my code, you can:

* **Train your model** by running **python train.py**
* **Evaluate mAP for COCO dataset** by running **python mAP_evaluation.py**
* **Test your model for COCO dataset** by running **python test_dataset.py**
* **Test your model for video** by running **python test_video.py**

## Experiments:

We trained our model by using 3 NVIDIA GTX 1080. Below is mAP (mean average precision) for COCO val2017 dataset 

|   Average Precision   |   IoU=0.50:0.95   |   area=   all   |   maxDets=100   |   0.314   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|   Average Precision   |      IoU=0.50     |   area=   all   |   maxDets=100   |   0.461   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|   Average Precision   |      IoU=0.75     |   area=   all   |   maxDets=100   |   0.343   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|   Average Precision   |   IoU=0.50:0.95   |   area= small   |   maxDets=100   |   0.093   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|   Average Precision   |   IoU=0.50:0.95   |   area= medium  |   maxDets=100   |   0.358   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|   Average Precision   |   IoU=0.50:0.95   |   area=  large  |   maxDets=100   |   0.517   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=1     |   0.268   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=10   |   0.382   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=100   |   0.403   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|     Average Recall    |   IoU=0.50:0.95   |   area= small   |   maxDets=100   |   0.117   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|     Average Recall    |   IoU=0.50:0.95   |   area= medium  |   maxDets=100   |   0.486   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|     Average Recall    |   IoU=0.50:0.95   |   area=  large  |   maxDets=100   |   0.625   |

## Results

Some output predictions for experiments for each dataset are shown below:

- **VOC2007**

<img src="demo/voc2007_1.jpg" width="280"> <img src="demo/voc2007_2.jpg" width="280"> <img src="demo/voc2007_3.jpg" width="280">

- **VOC2012**

<img src="demo/voc2012_1.jpg" width="280"> <img src="demo/voc2012_2.jpg" width="280"> <img src="demo/voc2012_3.jpg" width="280">

- **COCO2014**

<img src="demo/coco2014_1.jpg" width="280"> <img src="demo/coco2014_2.jpg" width="280"> <img src="demo/coco2014_3.jpg" width="280">

- **COCO2014+2017**

<img src="demo/coco2014_2017_1.jpg" width="280"> <img src="demo/coco2014_2017_2.jpg" width="280"> <img src="demo/coco2014_2017_3.jpg" width="280">
