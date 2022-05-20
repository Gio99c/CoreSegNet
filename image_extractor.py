import os
from PIL import Image

print(os.getcwd())

names = os.listdir("data/GTA5_modified/labels")
names.sort()

f = open("data/GTA5_modified/train.txt", "w+")


for name in names:
    #label_from_GTA5 = Image.open(f"data/GTA5/labels/"+name)
    #label_from_GTA5.save(f"data/GTA5_modified/labels/{name}")
    f.write(name+"\n")

f.close()

