import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage import io

import matplotlib.animation as ani
from IPython.display import HTML
import matplotlib

source_dir = Path('./data/source/test_img')
target_dir = Path('./results/target/test_latest/images')
label_dir = Path('./data/source/test_label_ori')

source_img_paths = sorted(source_dir.iterdir())
target_synth_paths = sorted(target_dir.glob('*synthesized*'))
target_label_paths = sorted(label_dir.iterdir())

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result/output.avi',fourcc, 20.0, (600,200))

for nframe in range(len(target_label_paths)):
    print(str(nframe)+'/'+str(len(target_label_paths)))

    source_img = cv2.imread(str(source_img_paths[nframe]))
    target_label = cv2.imread(str(target_label_paths[nframe]))
    target_synth = cv2.imread(str(target_synth_paths[nframe]))

    source_img = cv2.resize(source_img,(200,200))
    target_label = cv2.resize(target_label,(200,200))
    target_synth = cv2.resize(target_synth,(200,200))
    res=np.hstack((source_img,target_label))
    res=np.hstack((res,target_synth))
  
    cv2.imwrite("result/"+str(nframe)+".jpg",res)
    out.write(res)

out.release()
