import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class SiameseNet(nn.Module) :
	def __init__(self) :
		super(SiameseNet, self).__init__()
		self.conv_layer = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=10, stride=1, padding=0),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
			nn.ReLU())

		self.fc_layer = nn.Sequential(
			nn.Linear(256*6*6, 4096),
			nn.Sigmoid())

		self.last_layer = torch.nn.Sequential(
			torch.nn.Linear(4096, 1),
			torch.nn.Sigmoid())


	def forward(self, input_img_1, input_img_2) :
		conved1 = self.conv_layer(input_img_1)
		conved2 = self.conv_layer(input_img_2)

		hidden1 = self.fc_layer(torch.flatten(conved1, start_dim=1))
		hidden2 = self.fc_layer(torch.flatten(conved2, start_dim=1))

		prob = self.last_layer(torch.abs(hidden1 - hidden2))

		return prob


