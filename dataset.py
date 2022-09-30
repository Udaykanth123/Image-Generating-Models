import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import skimage.io
import numpy as np
import torchvision.transforms as transforms
from skimage.transform import resize


def normalize():
    return  transforms.Compose([
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])



class load_data(Dataset):
    def __init__(self,img_path):
        super(load_data,self).__init__()
        images_path=[g for g in os.listdir(img_path) if g.endswith(".png")]
        self.all_img_paths=[]
        for i in range(len(images_path)):
            self.all_img_paths.append(os.path.join(img_path,images_path[i]))
        self.normalise=normalize()
    
    def __len__(self):
        return len(self.all_img_paths)
    
    def __getitem__(self,id):
        img=skimage.io.imread(self.all_img_paths[id])/255
        img=resize(img,(64,64))
        # print(np.max(img))
        img=img.transpose(2, 0, 1)
        # img=np.expand_dims(img,axis=0)
        img=torch.from_numpy(img.astype(np.float32))
        img=self.normalise(img)
        # print(torch.min(img))
        return img


        
        
