# Object_Detection
Object detection with YOLO and OpenCV.

I used pretrained model of [YOLO v3](https://pjreddie.com/darknet/yolo/) for this project. This provides a simple web interface to the user, where they can upload a photo and perform Object Detection.

### Steps to run it on your system:


1:clone this repository using
>git clone https://github.com/Sonkaryasshu/Object_Detection.git

2:Recommended step *[Optional]*: [Create virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
>python3 -m venv env

Activate virtual environment 
>source env/bin/activate

3:Change directory to current project
>cd path/to/repo/Object_Detection

4:Install requirements
>pip install -r requirments.txt 

5:Run the app.py file
>python app.py

Now open localhost:5000 i.e. http://127.0.0.1:5000 and use as normal web application

### What is [YOLO](https://pjreddie.com/darknet/yolo/)?

You only look once, or YOLO, is one of the faster object detection algorithms out there. Though it is no longer the most accurate object detection algorithm, it is a very good choice when you need real-time detection, without loss of too much accuracy.

*As I can't explain whole YOLO here, you may want to understand it from the research paper*

*[YOLO](https://arxiv.org/pdf/1506.02640.pdf)*

*[YOLOv2](https://arxiv.org/pdf/1612.08242v1.pdf)*

*[YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)*

#### Overview of YOLO

YOLO (You Only Look Once), is a network for object detection. The object detection task consists in determining the location on the image where certain objects are present, as well as classifying those objects. Previous methods for this, like R-CNN and its variations, used a pipeline to perform this task in multiple steps. This can be slow to run and also hard to optimize, because each individual component must be trained separately. YOLO, does it all with a single neural network. From the paper:

>We reframe the object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.

So, to put it simple, you take an image as input, pass it through a neural network that looks similar to a normal CNN, and you get a vector of bounding boxes and class predictions in the output.

The first step to understanding YOLO is how it encodes its output. The input image is divided into an S x S grid of cells. For each object that is present on the image, one grid cell is said to be “responsible” for predicting it. That is the cell where the center of the object falls into.

Each grid cell predicts B bounding boxes as well as C class probabilities. The bounding box prediction has 5 components: (x, y, w, h, confidence). The (x, y) coordinates represent the center of the box, relative to the grid cell location (remember that, if the center of the box does not fall inside the grid cell, than this cell is not responsible for it). These coordinates are normalized to fall between 0 and 1. The (w, h) box dimensions are also normalized to [0, 1], relative to the image size.

#### Some advantages of YOLO:

- Speed (faster version (with smaller architecture) — 155 frames per sec but is less accurate)
- Predictions (object locations and classes) are made from one single network. Can be trained end-to-end to improve accuracy.
- YOLO is more generalized. It outperforms other methods when generalizing from natural images to other domains like artwork.
- Region proposal methods limit the classifier to the specific region. YOLO accesses to the whole image in predicting boundaries. With the additional context, YOLO demonstrates fewer false positives in background areas.
- YOLO detects one object per grid cell. It enforces spatial diversity in making predictions.
