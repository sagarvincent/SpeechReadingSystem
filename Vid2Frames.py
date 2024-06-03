import cv2
import numpy as np
import torch
from torchvision import transforms

# pscount -> no. of frames that should be considered per second 
#(pscount=3 => only 3 frames will taken from frames happening in a second)
def vid2frames(pscount,vpath,conv2gray):  
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    divfactor = 0 
    if(total_frames/fps > 30):
        divfactor = fps/pscount   

    frames = []

    fcount = 0
    while True:
        if(fcount%divfactor==0):
            ret, frame = cap.read()
            if not ret:
                break
            if conv2gray:    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            else:
                frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb) 

    # Convert the list of frames to a tensor
    frames_tensor = np.array(frames).transpose(0, 3, 1, 2)
    frames_tensor = frames_tensor / 255.0
    cap.release()
