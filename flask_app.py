# Created by YASHWANT SONKAR
# @SONKARyasshu

# import the necessary packages
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
import pickle
import cv2
import os
import uuid

app = Flask(__name__, static_folder='download')
UPLOAD_FOLDER = os.path.basename('upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER']= os.path.basename('download')

#setting default parameter(s) values 
default_confidence=0.5
default_threshold=0.3

@app.route('/')
def hello_world():
    return render_template('index.html')

def new_name(filename):
	nname=filename.split('.')
	return str(uuid.uuid4())+'.'+nname[-1]

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
# np.random.seed(5)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

@app.route('/upload',methods=['POST'])
def predict():
	image_in =request.files.get('image','')
	filename = os.path.join(app.config['UPLOAD_FOLDER'], image_in.filename)
	
	if os.path.exists(filename):
		nam=new_name(image_in.filename)
		os.rename(filename,os.path.join(app.config['UPLOAD_FOLDER'], nam))
		os.rename(os.path.join(app.config['DOWNLOAD_FOLDER'], image_in.filename) , os.path.join(app.config['DOWNLOAD_FOLDER'], nam))
		print("[File Renamed]")

	image_in.save(filename)
	print("[Image Saved In upload folder]")

	image = cv2.imread(filename)
	(H, W) = image.shape[:2]
	
	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	net.setInput(blob) 

	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > default_confidence:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, default_confidence,default_threshold)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

	print("[Detection Completed]")
	
	cv2.imwrite(os.path.join(app.config['DOWNLOAD_FOLDER'], image_in.filename), image)
	print("[Image Saved In download folder]")
	return redirect(url_for('download', filename=image_in.filename)) 

@app.route('/<filename>')
def download(filename):
    filename = os.path.join(app.config['DOWNLOAD_FOLDER'],filename)
    return render_template('index.html', filename = filename)

if __name__ == '__main__':
	app.run()
