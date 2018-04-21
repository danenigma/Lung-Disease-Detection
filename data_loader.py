import os
import numpy as np # linear algebra
import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torchvision
from torchvision import transforms
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt
from models import *
from PIL import Image
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import *

class XrayDataset(data.Dataset):
	""" Xray dataset """

	def __init__(self, data, labels, transform=None):
		super(XrayDataset, self).__init__()
	
		self.data   = data
		self.labels = labels
		self.transform = transform
	def __getitem__(self, index):
		"""
		returns 
		"""
				
		image  = torch.from_numpy(self.data[index]).transpose(0,2).float()
		target = torch.FloatTensor([int(self.labels[index])])  
		    
		if self.transform is not None:
			image = self.transform(image)

		return image, target

	def __len__(self):
		"""length of dataset"""
		return self.data.shape[0]


class OpenIDataset(data.Dataset):

	def __init__(self, data_dir='data', name = 'train', transform=None):
		self.table     = pd.read_pickle(os.path.join(data_dir,'{}_table.pkl'.format(name)))
		self.data_dir  = data_dir
		self.transform = transform

	def __len__(self):
		return len(self.table)
		
	def __getitem__(self, idx):
		img_name = self.table.iloc[idx, 0]
		fullname = os.path.join(self.data_dir, img_name+'.png')
		image  = Image.open(fullname).convert('RGB')
		if self.transform:
			image = self.transform(image)
			
		label  = int(self.table.iloc[idx, 2])
		
		return image, label

if __name__ == '__main__':
	
	data_dir    = 'data/images'

	img_size    = 224 

	train_trans = transforms.Compose([
		transforms.RandomResizedCrop(img_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	test_trans = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	val_ds     = OpenIDataset(data_dir, transform=train_trans, name='train')
	data_loader = data.DataLoader(val_ds, 
							 batch_size = 8,
							 shuffle = True)
							 
	
	for batch_index, (images, labels) in enumerate(data_loader):
		print('images: ', images.shape)
		print('labels: ', labels)
		break
		
	'''
	images = np.load('data/images.npy')
	labels = np.load('data/targets.npy')

	xrayds = XrayDataset(images, labels)
	batch_size = 8
	data_loader = data.DataLoader(
				  xrayds, batch_size = batch_size,
				  sampler = RandomSampler(xrayds))
	model = ResNetCNN(1)
	if torch.cuda.is_available():
		model.cuda()
		
	criterion = nn.BCELoss()
	params    = list(list(model.linear1.parameters()) + list(model.linear2.parameters()))
	optimizer = torch.optim.Adam(params, lr=1e-4)
	
	for batch_index, (images, labels) in enumerate(data_loader):
		images, labels = to_var(images), to_var(labels)
		model.zero_grad()
		out  = model(images)
		loss = criterion(out, labels)
		loss.backward()
		optimizer.step()
		print(loss)
	'''
