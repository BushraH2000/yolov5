import os
import subprocess


# convert SVG to PNG ..
import cairosvg # to download the library => https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
import urllib.request
import io
from PIL import Image

# Define the URL of the SVG image
svg_url = 'https://bitbtesting.000webhostapp.com/ssl.svg' #replace it with the image u want

# Download the SVG image from the URL
svg_data = urllib.request.urlopen(svg_url).read()

# Convert the SVG data to PNG data
png_data = cairosvg.svg2png(bytestring=svg_data)

# Convert the PNG data to an Image object
img = Image.open(io.BytesIO(png_data))

# Do something with the Image object, such as displaying it
img.show()

# call image analysis



os.system('cmd /k "python yolov5/detect.py --weights yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source img"') 
#os.system('cmd /k "python yolov5/detect.py --weights yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source gg.jpg"') 
##

#python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source ../PP.jpg

# Hit