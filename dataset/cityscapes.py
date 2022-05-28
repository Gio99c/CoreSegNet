import sys
sys.path.insert(1, "./")

import os
from pathlib import Path
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from utils import Map, MeanSubtraction, colorLabel, create_mask
from tqdm import tqdm




class Cityscapes(VisionDataset):
    def __init__(self, root, images_folder, labels_folder, train=True, info_file=None, transforms=[]):
        """
        Inputs:
            root: string, path of the root folder where images and labels are stored
            list_path: string, path of the file used to split the dataset (train.txt/val.txt)
            image_folder: string, path of the images folder
            labels_folder: string, path of the labels folder
            transform: transformation to be applied on the images
            target_transform: transformation to be applied on the labels

        self.images = list containing the paths of the images 
        self.labels = list contating the paths of the labels
        """
        super().__init__(root, transforms)
        
        self.train = train   
        self.list_path = "train.txt" if train else "val.txt"                              
        
        info = json.load(open(os.path.join(self.root, info_file)))
        
        self.mapper = dict(info["label2train"])
        self.mean = info["mean"]
        
        self.images_folder_path = Path(os.path.join(self.root, images_folder))    
        self.labels_folder_path = Path(os.path.join(self.root, labels_folder))    
        
        image_name_list = np.array(sorted(os.listdir(self.images_folder_path)))
        labels_list = np.array(sorted(os.listdir(self.labels_folder_path)))

        name_samples = [l.split("/")[1] for l in np.loadtxt(os.path.join(self.root, self.list_path), dtype="unicode")] 
        self.images = [img for img in image_name_list if str(img) in name_samples]    
        self.labels = [img for img in labels_list if str(img).replace("_gtFine_labelIds.png", "_leftImg8bit.png") in name_samples]  


    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return len(self.images)


    def __getitem__(self, index):
        image_path = os.path.join(self.images_folder_path,self.images[index])
        label_path = os.path.join(self.labels_folder_path,self.labels[index])

        image = np.array(Image.open(image_path), dtype=np.float32)
        label = np.array(Image.open(label_path), dtype=np.float32)

        image = MeanSubtraction(self.mean)(image)
        label = Map(self.mapper)(label)

        if self.transforms and self.train:
            seed = np.random.randint(10000)
            torch.manual_seed(seed)
            image = self.transforms(image)    
            torch.manual_seed(seed)
            label = self.transforms(label)   
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Resize((512, 1024))(image) 
            label = transforms.ToTensor()(label)
            label = transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST)(label)

        return image, label[0]


    def get_labels(self):
        labels = []
        for i in tqdm((range(self.__len__()))): 
            _, label = self.__getitem__(i)
            labels.append(label)
        return labels




if __name__ == "__main__":
    crop_width = 1024
    crop_height = 512
    composed = torchvision.transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop((crop_height, crop_width), pad_if_needed=True)])
    
    data = Cityscapes(root=os.path.join(os.getcwd(),"data","Cityscapes"), images_folder="images", labels_folder="labels", train=True, info_file="info.json") 
    
    image, label = data[0]
    
    #info
    info = json.load(open(os.path.join(os.getcwd(),"data","Cityscapes","info.json")))

    #image
    image = transforms.ToPILImage()(image.to(torch.uint8))
    image.show()

    #label
    palette = {i if i!=19 else 255:info["palette"][i] for i in range(20)}
    label = colorLabel(label,palette)
    label.show()