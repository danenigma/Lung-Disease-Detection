import os
import numpy as np # linear algebra
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.utils.data.sampler as sampler
from models import *
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


if __name__ == '__main__':
	
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
	
