Project 2 Deep Learning 
 
In this project we will build a deep learning model for classification. The model is a diagnostic 
model for the home-based COVID test. After taking the covid test, the model takes a picture of 
the antigen device to determine the result of the test.  
In this project, you need to use the object detection model that was developed in project 1. In 
this project, perform the following steps: 
1.  Use the object detection model to identify the COVID device. 
 
 
2.  Crop the image according to the boundary box. The new image contains mainly the COVID 
device. 
 
3.  Send the new image to the new AI model to determine the COVID test result 
a.  Two lines (T and C), positive test – you have COVID 
b.  One line (only C), negative test – you don’t have a COVID 
c.  No lines, the test is not valid 
 
In this project, you need to develop a new AI model for diagnostics, as described in item (3) 
above. The AI model generates the possible answers: 
  Yes: if the test is positive – a test is positive only if there are two lines on the device, the 
T-line and C-line 
  No: is the test is negative – a test is negative only if there is one line, the C-line 
  Invalid: if there are no lines 
Examples: 
Positive test result: 
 
Negative test result: 
 
 
Invalid 
 
 
You may use the same dataset, used in Project1. In addition, you can add more images, 
especially, to capture the “Invalid” state. 
 
 
 
GOOD LUCK! 
Adnan 