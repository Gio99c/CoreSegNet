import json
import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import random
import numbers
import torchvision
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.parameter_count import parameter_count
import matplotlib.pyplot as plt


#-------------------------------------------------------------
#-----------------PREDEFINED FUNCTIONS------------------------
#-------------------------------------------------------------

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=300, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power
	"""
	# if iter % lr_decay_iter or iter > max_iter:
	# 	return optimizer

	lr = init_lr*(1 - iter/max_iter)**power
	optimizer.param_groups[0]['lr'] = lr
	return lr
	# return lr

def get_label_info(csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	label = {}
	for iter, row in ann.iterrows():
		label_name = row['name']
		r = row['r']
		g = row['g']
		b = row['b']
		class_11 = row['class_11']
		label[label_name] = [int(r), int(g), int(b), class_11]
	return label

def one_hot_it(label, label_info):
	# return semantic_map -> [H, W]
	semantic_map = np.zeros(label.shape[:-1])
	for index, info in enumerate(label_info):
		color = label_info[info]
		# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
		equality = np.equal(label, color)
		class_map = np.all(equality, axis=-1)
		semantic_map[class_map] = index
		# semantic_map.append(class_map)
	# semantic_map = np.stack(semantic_map, axis=-1)
	return semantic_map

def one_hot_it_v11(label, label_info):
	# return semantic_map -> [H, W]
	semantic_map = np.zeros(label.shape[:-1])
	# from 0 to 11, and 11 means void
	class_index = 0
	for index, info in enumerate(label_info):
		color = label_info[info][:3]
		class_11 = label_info[info][3]
		if class_11 == 1:
			# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			# semantic_map[class_map] = index
			semantic_map[class_map] = class_index
			class_index += 1
		else:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			semantic_map[class_map] = 11
	return semantic_map

def one_hot_it_v11_dice(label, label_info):
	# return semantic_map -> [H, W, class_num]
	semantic_map = []
	void = np.zeros(label.shape[:2])
	for index, info in enumerate(label_info):
		color = label_info[info][:3]
		class_11 = label_info[info][3]
		if class_11 == 1:
			# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			# semantic_map[class_map] = index
			semantic_map.append(class_map)
		else:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			void[class_map] = 1
	semantic_map.append(void)
	semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
	return semantic_map

def reverse_one_hot(image):
	"""
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.
	# Arguments
		image: The one-hot format image
	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,1])

	# for i in range(0, w):
	#     for j in range(0, h):
	#         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
	#         x[i, j] = index
	image = image.permute(1, 2, 0)
	x = torch.argmax(image, dim=-1)
	return x

def compute_global_accuracy(pred, label):
	pred = pred.flatten()
	label = label.flatten()
	total = len(label)
	count = 0.0
	for i in range(total):
		if pred[i] == label[i]:
			count = count + 1.0
	return float(count) / float(total)

def fast_hist(a, b, n):
	'''
	a and b are predict and mask respectively
	n is the number of classes
	'''
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
	epsilon = 1e-5
	return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def cal_miou(miou_list, csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	miou_dict = {}
	cnt = 0
	for iter, row in ann.iterrows():
		label_name = row['name']
		class_11 = int(row['class_11'])
		if class_11 == 1:
			miou_dict[label_name] = miou_list[cnt]
			cnt += 1
	return miou_dict, np.mean(miou_list)

class OHEM_CrossEntroy_Loss(nn.Module):
	def __init__(self, threshold, keep_num):
		super(OHEM_CrossEntroy_Loss, self).__init__()
		self.threshold = threshold
		self.keep_num = keep_num
		self.loss_function = nn.CrossEntropyLoss(reduction='none')

	def forward(self, output, target):
		loss = self.loss_function(output, target).view(-1)
		loss, loss_index = torch.sort(loss, descending=True)
		threshold_in_keep_num = loss[self.keep_num]
		if threshold_in_keep_num > self.threshold:
			loss = loss[loss>self.threshold]
		else:
			loss = loss[:self.keep_num]
		return torch.mean(loss)

def group_weight(weight_group, module, norm_layer, lr):
	group_decay = []
	group_no_decay = []
	for m in module.modules():
		if isinstance(m, nn.Linear):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
			if m.weight is not None:
				group_no_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)

	assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
	weight_group.append(dict(params=group_decay, lr=lr))
	weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
	return weight_group

def map_label(label, label_mappign_info):
    # re-assign labels to match the format of Cityscapes
    label_copy = 255 * np.ones(label.shape, dtype=np.float32)
    for k, v in label_mappign_info.items():
            label_copy[label == k] = v

    return label_copy

def one_hot(label):
    # return semantic_map -> [H, W, class_num]
    semantic_map = []
    for class_index in range(20):
        if class_index == 19:
            class_index = 255
        
        mask = label==class_index
        semantic_map.append(mask)
    
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)

    return semantic_map



#------------------------------------------------------------
#------------------CUSTOM FUNCTIONS--------------------------
#------------------------------------------------------------

def colorLabel(label, palette):
    composed = torchvision.transforms.Compose([ToNumpy(), Map2(palette), transforms.ToTensor(), transforms.ToPILImage()])
    label = composed(label)
    return label

def parameter_flops_count(model, discriminator, input=torch.randn(8, 3, 512, 1024)): 
	# return the count of discirminator's flops and parameters 
    flops = FlopCountAnalysis(discriminator, F.softmax(model(input)[0])) 
    parameters = sum(parameter_count(discriminator).values())
    return (flops, parameters)

def save_images(mean, palette, image, predict, label, path_to_save):
	#Save an output examples
	#image
	image = image[0].clone().detach()
	image = (image.permute(1, 2, 0) + mean).permute(2, 0, 1)
	image = transforms.ToPILImage()(image.to(torch.uint8))
	#prediction
	predict = torch.tensor(predict.copy(), dtype=torch.uint8)
	predict = colorLabel(predict, palette) 
	#label from np to Pil Image
	label = torch.tensor(label.copy(), dtype=torch.uint8)
	label = colorLabel(label, palette)
	#create the figure
	fig, axs = plt.subplots(1,3, figsize=(10,5))
	axs[0].imshow(image)
	axs[0].axis('off')
	axs[1].imshow(predict)
	axs[1].axis('off')
	axs[2].imshow(label)
	axs[2].axis('off')
	#save the final result
	plt.savefig(path_to_save) #@ Salvare in png | che significa i/2 ? | le immagini vengono sovrascritte ad ogni epoch? | Aggiungere la directory negli argomenti

def get_index(i):
	"""
	Create the index to save the example
	"""
	return "0"*(3-len(str(i)))+str(i)

#------------------------------------------------------------
#------------------CUSTOM TRANSFORMS-------------------------
#------------------------------------------------------------

class Map:
    """
    Maps every pixel to the respective object in the dictionary
    Input:
        mapper: dict, dictionary of the mapping
    """
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, input):
        return np.vectorize(self.mapper.__getitem__, otypes=[np.float32])(input)

class Map2:
    """
    Maps every pixel to the respective object in the dictionary
    Input:
        mapper: dict, dictionary of the mapping
    """
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, input):
        return np.array([[self.mapper[element] for element in row]for row in input], dtype=np.float32)

class ToTensor:
    """
    Convert into a tensor of float32: differently from transforms.ToTensor() this function does not normalize the values in [0,1] and does not swap the dimensions
    """
    def __call__(self, input):
        return torch.as_tensor(input, dtype=torch.float32)

class ToNumpy:
    """
    Convert into a tensor into a numpy array
    """
    def __call__(self, input):
        return input.numpy()

# Don't know if it will be useful or if we will subtract the mean inside the dataset class
class MeanSubtraction:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, input):
        return input - self.mean

class RandomCrop(object):
	"""Crop the given PIL Image at a random location.
	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
		pad_if_needed (boolean): It will pad the image if smaller than the
			desired size to avoid raising an exception.
	"""

	def __init__(self, size, seed, padding=0, pad_if_needed=False):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding
		self.pad_if_needed = pad_if_needed
		self.seed = seed

	@staticmethod
	def get_params(img, output_size, seed):
		"""Get parameters for ``crop`` for a random crop.
		Args:
			img (PIL Image): Image to be cropped.
			output_size (tuple): Expected output size of the crop.
		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
		random.seed(seed)
		w, h = img.size
		th, tw = output_size
		if w == tw and h == th:
			return 0, 0, h, w
		i = random.randint(0, h - th)
		j = random.randint(0, w - tw)
		return i, j, th, tw

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped.
		Returns:
			PIL Image: Cropped image.
		"""
		if self.padding > 0:
			img = torchvision.transforms.functional.pad(img, self.padding)

		# pad the width if needed
		if self.pad_if_needed and img.size[0] < self.size[1]:
			img = torchvision.transforms.functional.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
		# pad the height if needed
		if self.pad_if_needed and img.size[1] < self.size[0]:
			img = torchvision.transforms.functional.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

		i, j, h, w = self.get_params(img, self.size, self.seed)

		return torchvision.transforms.functional.crop(img, i, j, h, w)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)