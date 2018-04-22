import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class ResNetCNN(nn.Module):

	def __init__(self):
		"""Load the pretrained ResNet-152 and replace top fc layer."""
		super(ResNetCNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-1]      # delete the last fc layer.
		self.resnet  = nn.Sequential(*modules)
		self.linear1 = nn.Linear(resnet.fc.in_features, 1024)
		self.linear2 = nn.Linear(1024, 1024)    
		self.linear3 = nn.Linear(1024, 2)
		
		self.init_weights()
		
	def init_weights(self):
		"""Initialize the weights."""
		self.linear1.weight.data.normal_(0.0, 0.02)
		self.linear1.bias.data.fill_(0)

		self.linear2.weight.data.normal_(0.0, 0.02)
		self.linear2.bias.data.fill_(0)

		prop_abnormal = 0.635
		data_dist = torch.FloatTensor([1-prop_abnormal, prop_abnormal])

		self.linear3.weight.data.normal_(0.0, 0.02)
		self.linear3.bias.data = data_dist #fill_(0)

		
	def forward(self, images):
		"""Extract the image feature vectors."""
		features = self.resnet(images)
		print('feats: ', features.shape)
		features = Variable(features.data)
		features = features.view(features.size(0), -1)
		features = F.relu(self.linear1(features))
		features = F.relu(self.linear2(features))
		features = F.relu(self.linear3(features))

		return features
class DenseNet121(nn.Module):

	def __init__(self):
		"""Load the pretrained DenseNet-121 and replace top fc layer."""
		super(DenseNet121, self).__init__()
		densenet = models.densenet121(pretrained=True)
		#print(densenet)
		modules  = list(densenet.children())[:-1]      # delete the last fc layer.

		self.densenet= nn.Sequential(*modules)
		#print(self.densenet)
		self.avgpool = nn.AvgPool2d(kernel_size=7)
		self.linear1 = nn.Linear(densenet.classifier.in_features, 1024)
		self.linear2 = nn.Linear(1024, 1024)    
		self.linear3 = nn.Linear(1024, 2)
		
		self.init_weights()
		
	def init_weights(self):
		"""Initialize the weights."""
		self.linear1.weight.data.normal_(0.0, 0.02)
		self.linear1.bias.data.fill_(0)

		self.linear2.weight.data.normal_(0.0, 0.02)
		self.linear2.bias.data.fill_(0)

		prop_abnormal = 0.635
		data_dist = torch.FloatTensor([1-prop_abnormal, prop_abnormal])

		self.linear3.weight.data.normal_(0.0, 0.02)
		self.linear3.bias.data = data_dist #fill_(0)

		
	def forward(self, images):
		"""Extract the image feature vectors."""
		features = self.densenet(images)
		features = Variable(features.data)
		features = self.avgpool(features)
		features = features.view(features.size(0), -1)
		features = F.relu(self.linear1(features))
		features = F.relu(self.linear2(features))
		features = F.relu(self.linear3(features))

		return features

if __name__=='__main__':
	resnet = ResNetCNN()
	densenet = DenseNet121()
	#print(densenet)
	x_test = torch.randn(4,3,224,224)
	print(densenet(Variable(x_test)))
	
