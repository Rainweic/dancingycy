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
import math

relationship=[]
relationship.append([15,17])
relationship.append([15,0])
relationship.append([0,16])
relationship.append([16,18])
relationship.append([0,1])
relationship.append([2,1])
relationship.append([5,1])
relationship.append([8,1])
relationship.append([2,3])
relationship.append([3,4])
relationship.append([5,6])
relationship.append([6,7])
relationship.append([8,9])
relationship.append([9,10])
relationship.append([10,11])
relationship.append([11,22])
relationship.append([22,23])
relationship.append([11,24])
relationship.append([8,12])
relationship.append([12,13])
relationship.append([13,14])
relationship.append([14,21])
relationship.append([14,19])
relationship.append([20,19])
relationship_hand=[]
relationship_hand.append([0,1])
relationship_hand.append([1,2])
relationship_hand.append([2,3])
relationship_hand.append([3,4])
relationship_hand.append([0,5])
relationship_hand.append([5,6])
relationship_hand.append([6,7])
relationship_hand.append([7,8])
relationship_hand.append([0,9])
relationship_hand.append([9,10])
relationship_hand.append([10,11])
relationship_hand.append([11,12])
relationship_hand.append([0,13])
relationship_hand.append([13,14])
relationship_hand.append([14,15])
relationship_hand.append([15,16])
relationship_hand.append([0,17])
relationship_hand.append([17,18])
relationship_hand.append([18,19])
relationship_hand.append([19,20])

def getLabel(pose,left,right):
    label = np.zeros((512,512), dtype=np.uint8)
    global relationship
    global relationship_hand
    for i in range(24):
        #print(str(i)+" "+str(pose[i]))
        if pose[relationship[i][0]][2]==0 or pose[relationship[i][1]][2]==0:
            continue
        joint_coords = [pose[relationship[i][0]][:2],pose[relationship[i][1]][:2]]
        #print(joint_coords)
        coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
        limb_dir = joint_coords[0] - joint_coords[1]
        limb_length = np.linalg.norm(limb_dir)
        angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(label, polygon, i+1)
    for i in range(20):
        if left[relationship_hand[i][0]][2]==0 or left[relationship_hand[i][1]][2]==0:
            continue
        joint_coords = [left[relationship_hand[i][0]][:2],left[relationship_hand[i][1]][:2]]
        triangle = np.array([joint_coords[0], joint_coords[1] , joint_coords[1]], np.int32)
        cv2.fillConvexPoly(label, triangle, 25)
        #print(joint_coords)
        # coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
        # limb_dir = joint_coords[0] - joint_coords[1]
        # limb_length = np.linalg.norm(limb_dir)
        # angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        # polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
        # cv2.fillConvexPoly(label, polygon, i+25)
    for i in range(20):
        if right[relationship_hand[i][0]][2]==0 or right[relationship_hand[i][1]][2]==0:
            continue
        joint_coords = [right[relationship_hand[i][0]][:2],right[relationship_hand[i][1]][:2]]
        triangle = np.array([joint_coords[0], joint_coords[1], joint_coords[1]], np.int32)
        cv2.fillConvexPoly(label, triangle, 26)
        #print(joint_coords)
        # coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
        # limb_dir = joint_coords[0] - joint_coords[1]
        # limb_length = np.linalg.norm(limb_dir)
        # angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        # polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
        # cv2.fillConvexPoly(label, polygon, i+45)
    #print("_________________")
    return label

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
params["hand"] = True
params["disable_blending"] = True
params["number_people_max"] = 1

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

train_label_dir = train_dir.joinpath('train_label_show')
train_label_dir.mkdir(exist_ok=True)

pose_cords = []
last_cord = [0,0]
start = time.time()
last_label = []

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
    #label = cv2.cvtColor(datum.cvOutputData,cv2.COLOR_RGB2GRAY)

    #print(datum.poseKeypoints)
    crop_size=25
    try:
        head_cord=[int(datum.poseKeypoints[0,0,0]),int(datum.poseKeypoints[0,0,1])]
        #print(datum.handKeypoints[0][0])
        #print(datum.handKeypoints[1][0])
        label = getLabel(datum.poseKeypoints[0],datum.handKeypoints[0][0],datum.handKeypoints[1][0])
    except Exception as e:
        print(e)
        head_cord=last_cord
        label = last_label
    last_cord=head_cord
    last_label=label
    #print(head_cord)
    if head_cord[1]<crop_size:
        head_cord[1]=crop_size
    if head_cord[1]+crop_size>=512:
        head_cord[1]=512-crop_size-1
    if head_cord[0]<crop_size:
        head_cord[0]=crop_size
    if head_cord[0]+crop_size>=512:
        head_cord[0]=512-crop_size-1
    pose_cords.append(head_cord)
    head = img[int(head_cord[1] - crop_size): int(head_cord[1] + crop_size),
            int(head_cord[0] - crop_size): int(head_cord[0] + crop_size), :]
            
    label[:,:,0]=np.mod(label[:,:,0]*509,256)
    label[:,:,1]=np.mod(label[:,:,1]*3467,256)
    label[:,:,2]=np.mod(label[:,:,2]*15031,256)
    cv2.imwrite(str(train_label_dir.joinpath('{:05}.png'.format(idx))), label)

end = time.time()
print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
torch.cuda.empty_cache()
