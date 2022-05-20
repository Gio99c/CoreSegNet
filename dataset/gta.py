import sys
sys.path.insert(1, "./")
import os

from turtle import color
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision
import json
from tqdm import tqdm

from utils import Map, Map2, MeanSubtraction, ToNumpy, colorLabel, create_mask, one_hot




class GTA(VisionDataset):
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
        self.mean = info["mean"]
        self.images_folder_path = Path(os.path.join(self.root, images_folder))     # absolute path of the folder containing the images
        self.labels_folder_path = Path(os.path.join(self.root, labels_folder))     # absolute path of the folder containing the labels
        
        #Retrive the file names of the images and labels contained in the indicated folders
        image_name_list = np.array(sorted(os.listdir(self.images_folder_path)))
        labels_list = np.array(sorted(os.listdir(self.labels_folder_path)))

        #Prepare lists of data and labels
        name_samples = list(np.loadtxt(os.path.join(self.root, self.list_path), dtype="unicode")) # creates the list of the images names for the train/validation according to list_path
        self.images = [img for img in image_name_list if str(img) in name_samples]    # creates the list of images names filtered according to name_samples
        self.labels = [img for img in labels_list if str(img) in name_samples]  # creates the list of label image names filtered according to name_samples


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
        
        if self.transforms:
            seed = np.random.randint(10000)
            torch.manual_seed(seed)
            image = self.transforms(image)    # applies the transforms for the images
            torch.manual_seed(seed)
            label = self.transforms(label)    # applies the transforms for the labels
        else: 
            image = transforms.ToTensor()(image)
            image = transforms.Resize((720, 1280))(image) 
            label = transforms.ToTensor()(label)
            label = transforms.Resize((720, 1280), interpolation=transforms.InterpolationMode.NEAREST)(label)

        return image, label[0]
    
    def get_labels(self):
        print('getting labels')
        labels = []
        for i in tqdm((range(self.__len__()))): 
            _, label = self.__getitem__(i)
            labels.append(label)
        return labels




if __name__ == "__main__":
    crop_width = 1280
    crop_height = 720
    composed = torchvision.transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop((crop_height, crop_width), pad_if_needed=True)])
    data = GTA(root=os.path.join(os.getcwd(),"data","GTA5"), images_folder="images", labels_folder="labels", list_path="train.txt", info_file="info.json") 
    image, label = data[38]



    #print(image.size())

    # #info
    info = json.load(open(os.path.join(os.getcwd(),"data","GTA5","info.json")))

    # # #Image
    image = transforms.ToPILImage()(image.to(torch.uint8))

    from PIL import ImageFont
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    
    # draw.text((x, y),"Sample Text",(r,g,b))
    
    image.show(title="Your Title Here")
    
    # # #Label
    palette = {i if i!=19 else 255:info["palette"][i] for i in range(20)}

    layer_0 = one_hot(label)[0][16]
    layer_0 = colorLabel(layer_0, palette)

    #layer_0.show()

    label = colorLabel(label,palette)
    #label.show()
    #image.show()
    
    


    # #get_labels Test
    # labels = data.get_labels()
    # print(len(labels))
    # label = colorLabel(labels[0], palette)

    # fig, axs = plt.subplots(1,2, figsize=(10,5))
    # axs[0].imshow(image)
    # axs[0].axis('off')
    # axs[1].imshow(label)
    # axs[1].axis('off')

    # plt.show()

    mask,_ = create_mask(data.get_labels())
    print(mask.shape)
    for i,layer in enumerate(mask):
        
        layer_1 = layer
        print("layer shape: ",layer_1.shape)
        layer_1 = torch.tensor(layer_1.clone().detach()*255, dtype=torch.uint8)
        layer_1 = transforms.ToPILImage()(layer_1)
        draw = ImageDraw.Draw(layer_1)
        draw.text((0, 0),f"Etichetta numero {i}",fill =128)
        layer_1.show()

    # #check prob
    # prob = torch.sum(mask, axis = -3)
    # print(prob)
    






    
    #transforms.ToPILImage()(image_label.to(torch.uint8)).show()
