Phases for Project 1  
Phase 1: Data Collection 
  Collect images (JPG format) of COVID-19 devices 
  At least 400 images 
  Divide the data into train and test datasets 
o  For example, 75% and 25% or 80% + 20%, etc. 
 
Phase 2: Object Detection 
  Boundary box labeling  
  Investigate and use an appropriate tool 
o  Examples: LabelMe or Roboflow 
o  Labelling for any position of the object 
o  [optional] Label as polygon with four points in the space 
  Label the images – only the train set 
  My recommendation is CVAT: 
o  On the web site cvat.ai 
o  Free download using cvat container 
 
Phase 3: Training the object detection model 
  Investigate and choose the AI model for object detection 
o  e.g., YOLOv5, SSD, Faster R-CNN, YOLO8 
  Goto https://huggingface.co/ for more models 