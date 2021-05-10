import torch
import cv2
# the working directory will be inside the final GAN directory
# need to install necessary requirements for yolov5
!pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt


input_class = 'skirt' # input provided by the user
model1 = ['shirt','tshirt']
model2 = ['skirt','short']
model3 = ['pant','jeans']
model4 = ['Solid straight kurtas','Solid straight kurtis', 'Printed kurtas','Printed kurtis', 'Sherwani', 'Anarkali', 'Banarasi Saree']

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolo/best.pt') 


# Images generated from ensemble_images are added here for detection.
img = ['./ensemble_images/netG_210/short/short1000_  0.png','./ensemble_images/netG_210/skirt/skirt1000_  0.png','./ensemble_images/netG_210/skirt/jeans1000_  0.png']

# Inference
results = model(img)

# the class of each image is compared with input class
#if there are more than same predictions, we pick one with higher confidence score
list_results = []
list_conf = []
for i in range(len(img)):
  a = (results.pandas().xyxy[i]).loc[:,'name'][0]
  conf = (results.pandas().xyxy[i]).loc[:,'confidence'][0]
  if(a==input_class):
    list_results.append(i)
    list_conf.append(conf)

max_val = max(list_conf)
max_index = list_conf.index(max_val)
index = list_results[max_index]
print(index)
# final image is saved as max.png
cv2.imwrite("./max.png",cv2.cvtColor(img[index], cv2.COLOR_RGB2BGR))