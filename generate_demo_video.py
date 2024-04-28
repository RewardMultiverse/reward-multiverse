import cv2
import os

# Directory containing images
image_folder = 'demo/bear'
# Video file to save to
video_name = 'bear.mov'

images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
images.sort()  # Sort the images by name (or modify to sort by desired criteria)

# Obtain frame size from the first image
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
# The 'DIVX' codec works well on Windows. Use 'XVID' for Linux or MacOS.
# fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter(video_name, fourcc, 3, (width, height))  # 1 is the frame rate

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
video.release()
