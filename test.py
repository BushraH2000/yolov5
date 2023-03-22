import os
import subprocess
os.system('cmd /k "python yolov5/detect.py --weights yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source HH.jpg"') 
#os.system('cmd /k "python yolov5/detect.py --weights yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source gg.jpg"') 


#python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source ../PP.jpg

