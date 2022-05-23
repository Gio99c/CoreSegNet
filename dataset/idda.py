import sys
sys.path.insert(1, "./")

from tqdm import tqdm
import os
from pathlib import Path
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from utils import Map, colorLabel, create_mask




class IDDA(VisionDataset):
    def __init__(self, root, images_folder, labels_folder, list_path, info_file=None, transforms=[]):
        """
        Inputs:
            root: string, path of the root folder where images and labels are stored
            list_path: string, path of the file used to split the dataset (train.txt/val.txt)
            image_folder: string, path of the images folder
            labels_folder: string, path of the labels folder
            transform: transformation to be applied on the images

        self.images = list containing the paths of the images 
        self.labels = list contating the paths of the labels
        """
        super().__init__(root, transforms)

        self.list_path = list_path
        
        info = json.load(open(os.path.join(self.root, info_file)))         
        self.mapper = dict(info["label2train"])
        
        self.images_folder_path = Path(os.path.join(self.root, images_folder))    
        self.labels_folder_path = Path(os.path.join(self.root, labels_folder))    
        
        image_name_list = np.array(sorted(os.listdir(self.images_folder_path)))
        labels_list = np.array(sorted(os.listdir(self.labels_folder_path)))

        name_samples = list(np.loadtxt(os.path.join(self.root, self.list_path), dtype="unicode")) 
        self.images = [img for img in image_name_list if str(img) in name_samples]    
        self.labels = [img for img in labels_list if str(img) in name_samples]  


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
        
        label = Map(self.mapper)(label)
        
        if self.transforms:
            seed = np.random.randint(10000)
            torch.manual_seed(seed)
            image = self.transforms(image)    
            torch.manual_seed(seed)
            label = self.transforms(label)    
        else: 
            image = transforms.ToTensor()(image)
            image = transforms.Resize((540, 960))(image) 
            label = transforms.ToTensor()(label)
            label = transforms.Resize((540, 960), interpolation=transforms.InterpolationMode.NEAREST)(label)

        return image, label[0]
    

    def get_labels(self):
        print('getting labels')
        labels = []
        for i in tqdm((range(self.__len__()))): 
            _, label = self.__getitem__(i)
            labels.append(label)
        return labels




if __name__ == "__main__":
    crop_width = 960
    crop_height = 540
    composed = torchvision.transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop((crop_height, crop_width), pad_if_needed=True)])
    
    data = IDDA(root=os.path.join(os.getcwd(),"data","IDDA"), images_folder="images", labels_folder="labels", list_path="train.txt", info_file="info.json") 
    
    image, label = data[0]

    #info
    info = json.load(open(os.path.join(os.getcwd(),"data","GTA5","info.json")))

    #image
    image = transforms.ToPILImage()(image.to(torch.uint8))
    image.show()
    
    #label
    palette = {i if i!=19 else 255:info["palette"][i] for i in range(20)}
    label = colorLabel(label, palette)
    label.show()

    #create mask and print
    from PIL import ImageDraw
    mask,_ = create_mask(data.get_labels())
    print(mask.shape)
    for i,layer in enumerate(mask):
        layer = torch.tensor(layer.clone().detach()*255, dtype=torch.uint8)
        layer = transforms.ToPILImage()(layer)
        draw = ImageDraw.Draw(layer)
        draw.text((0, 0),f"Etichetta numero {i}",fill =128)
        layer.show()

    #check prob (it does not sum to 1 if mask has ones filled layers)
    prob = torch.sum(mask, axis = -3)
    print(prob)