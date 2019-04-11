import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import os
import warnings
import sys
from sys import platform
import argparse
import time

save_dir = Path('./data/target/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

if len(os.listdir('./data/target/images'))<100:
    cap = cv2.VideoCapture(str(save_dir.joinpath('mv.mp4')))
    i = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if flag == False :
            break
        cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(i))), frame)
        if i%100 == 0:
            print('Has generated %d picetures'%i)
        i += 1

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/home/bhrgzn/openpose/build/python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/home/bhrgzn/openpose/build/x64/Release;' +  dir_path + '/home/bhrgzn/openpose/build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('/home/bhrgzn/openpose/build/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="/home/bhrgzn/openpose/examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/home/bhrgzn/openpose/models/"
params["face"] = True
params["disable_blending"] = True

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# '''make label images for pix2pix'''
train_dir = save_dir.joinpath('train')
train_dir.mkdir(exist_ok=True)

train_img_dir = train_dir.joinpath('train_img')
train_img_dir.mkdir(exist_ok=True)
train_label_dir = train_dir.joinpath('train_label')
train_label_dir.mkdir(exist_ok=True)
train_head_dir = train_dir.joinpath('head_img')
train_head_dir.mkdir(exist_ok=True)

pose_cords = []
start = time.time()
last_cord = [0,0]
for idx in tqdm(range(len(os.listdir(str(img_dir))))):
    img_path = img_dir.joinpath('{:05}.png'.format(idx))
    img = cv2.imread(str(img_path))
    shape_dst = np.min(img.shape[:2])
    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2
  
    shape_dst2 = np.max(img.shape[:2])
    img2=np.zeros((512,512,3), dtype=np.uint8)
    if img.shape[0]>img.shape[1]:
        img = cv2.resize(img, (img.shape[1]*512//img.shape[0],512))
        save=(512-img.shape[1])//2
        img2[:img.shape[0],save:save+img.shape[1]]=img
    else:
        img = cv2.resize(img, (512,img.shape[0]*512//img.shape[1]))
        save=(512-img.shape[0])//2
        img2[save:save+img.shape[0],:img.shape[1]]=img
    
    img=np.copy(img2)

    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    
    crop_size=25
    try:
        head_cord=[int(datum.faceKeypoints[0,0,0]),int(datum.faceKeypoints[0,0,1])]
    except:
        head_cord=last_cord
    last_cord=head_cord
    #print(head_cord)
    if head_cord[1]<crop_size:
        head_cord[1]=crop_size
    if head_cord[1]+crop_size>=512:
        head_cord[1]=512-crop_size-1
    if head_cord[0]<crop_size:
        head_cord[0]=crop_size
    if head_cord[0]+crop_size>=512:
        head_cord[0]=512-crop_size-1
    head = img[int(head_cord[1] - crop_size): int(head_cord[1] + crop_size),
            int(head_cord[0]): int(head_cord[0] + crop_size*2), :]
    plt.imshow(head)
    plt.savefig(str(train_head_dir.joinpath('pose_{}.jpg'.format(idx))))
    plt.clf()

# np.save(str((save_dir.joinpath('pose.npy'))), pose_cords)
end = time.time()
print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")

pose_cords = np.array(pose_cords, dtype=np.int)
np.save(str((save_dir.joinpath('pose.npy'))), pose_cords)
torch.cuda.empty_cache()
