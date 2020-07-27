import torch
import torchvision
import torchvision.transforms as transforms
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SiameseDataset(Dataset):
	def __init__(self, alphabet_path_list, size, distort=False) :
		'''
			path_list : list of path of alphabet_folders to train;
			size : size of dataset
		'''
		self.alphabet_path_list = alphabet_path_list
		self.size = size
		self.distort = distort
		self.pairs = []
		self.labels = []
		# self.transforms = [] # what transform to do...

		for data_iter in range(0, self.size) :
			# randomly select alphabet; and its character
			alphabet_path = random.choice(alphabet_path_list)
			char_list = os.listdir(alphabet_path)

			char_path_1 = self.pop_subpath(char_list, alphabet_path)

			# choose to add whther same or different pair
			self.labels.append(random.getrandbits(1))
			is_same = bool(self.labels[-1])

			char_path_2 = char_path_1 if is_same \
							else self.pop_subpath(char_list, alphabet_path)

			# pop image's path from each character
			image_list_1 = os.listdir(char_path_1)
			image_path_1 = self.pop_subpath(image_list_1, char_path_1)
			image_list_2 = os.listdir(char_path_2)
			image_path_2 = self.pop_subpath(image_list_2, char_path_2)

			self.pairs.append((image_path_1, image_path_2))


	def pop_subpath(self, listdir, base_path, pop_index=None) :
		if pop_index is None :
			pop_index = random.randrange(len(listdir))

		popped = listdir.pop(pop_index)

		return os.path.join(base_path, popped)

	def __getitem__(self, index) :
		image_path_1, image_path_2 = self.pairs[index]
		label = self.labels[index]

		image_1 = Image.open(image_path_1).convert('1')
		image_2 = Image.open(image_path_2).convert('1')

		return image_1, image_2, label

def get_loader(root, batch_size, dataset_size, shuffle) :
	alphabet_path_list = [os.path.join(root, alphabet) for alphabet in os.listdir(root)]

	siamese = SiameseDataset(alphabet_path_list, dataset_size)

	data_loader = DataLoader(dataset=siamese,
							batch_size=batch_size,
							shuffle=shuffle)

	return data_loader