import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class ResNetCNN(nn.Module):

    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNetCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear1 = nn.Linear(resnet.fc.in_features, 1024)
        self.linear2 = nn.Linear(1024, embed_size)
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear1.weight.data.normal_(0.0, 0.02)
        self.linear1.bias.data.fill_(0)

        self.linear2.weight.data.normal_(0.0, 0.02)
        self.linear2.bias.data.fill_(0)

        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.linear2(self.linear1(features))
        return F.sigmoid(features)

if __name__=='__main__':
	resnet = ResNetCNN(1)
	print(resnet)
	x_test = torch.randn(8,3,224,224)
	print(resnet(Variable(x_test)))
	
