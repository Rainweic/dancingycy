import model
import dataset
import cv2
from trainer import Trainer
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.io import imsave
from imageio import get_writer
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])
    
device = torch.device('cuda')


def load_models(directory):
    generator = model.GlobalGenerator(n_downsampling=2, n_blocks=6)
    gen_name = os.path.join(directory, 'final_generator.pth')

    if os.path.isfile(gen_name):
        gen_dict = torch.load(gen_name)
        generator.load_state_dict(gen_dict)
        
    return generator.to(device)
    
def torch2numpy(tensor):
        generated = tensor.detach().cpu().permute(1, 2, 0).numpy()
        generated[generated < -1] = -1
        generated[generated > 1] = 1
        generated = (generated + 1) / 2 * 255
        return generated.astype(np.uint8)
    
def get_full_sample(fake_img, head_pos):
        crop_size = 48
        size = fake_img.shape[:-1]
        left = int(head_pos[0] - crop_size / 2)  # don't suppose left will go out of bound eh?
        left = left if left >= 0 else 0
        left = size[1] - crop_size if left + crop_size > size[1] else left

        top = int(head_pos[1] - crop_size / 2)
        top = top if top >= 0 else 0
        top = size[0] - crop_size if top + crop_size > size[0] else top

        fake_head = image_transforms(fake_img[top: top + crop_size, left: left + crop_size,  :])
        
        return fake_head, top, top + crop_size, left, left + crop_size

    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    dataset_dir = '../data/face'   # save test_sync in this folder
    pose_name = '../data/source/pose_source_norm.npy' # coordinate save every heads
    ckpt_dir = '../checkpoints/face'
    result_dir = './results'
    save_dir = '../results/full_fake/'
    
    target_dir = Path('../results/target/test_latest/images')
    target_synth_paths = sorted(target_dir.glob('*synthesized*'))

    if not os.path.exists(save_dir):
        print('generate %s'%save_dir)
        os.mkdir(save_dir)
    else:
        print(save_dir, 'is existing...')


    image_folder = dataset.ImageFolderDataset(dataset_dir, cache=os.path.join(dataset_dir, 'local.db'), is_test=True)
    face_dataset = dataset.FaceCropDataset(image_folder, pose_name, image_transforms, crop_size=48)
    pose = np.load(pose_name)
    length = len(face_dataset)
    print('Picture number',len(target_synth_paths))

    generator = load_models(os.path.join(ckpt_dir))

    for i in tqdm(range(len(target_synth_paths))):#length
        print(pose[i])
        fake_full = cv2.imread(str(target_synth_paths[i]))
        fake_head, top, bottom, left, right = get_full_sample(fake_full, pose[i])

        with torch.no_grad():
            fake_head.unsqueeze_(0)
            fake_head = fake_head.to(device)
            residual = generator(fake_head)
            enhanced = fake_head + residual

        enhanced.squeeze_()
        enhanced = torch2numpy(enhanced)
        fake_full_old = fake_full.copy()
        fake_full[top: bottom, left: right, :] = enhanced

        b, g, r = cv2.split(fake_full)
        fake_full = cv2.merge([r, g, b])
        cv2.imwrite(save_dir+ '{:05}.png'.format(i),fake_full[:,:,::-1])

