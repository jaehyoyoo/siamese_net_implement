import torch
import torchvision
import torchvision.transforms as transforms
import os
import random


from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SiameseDataset(Dataset):
	def __init__(self, alphabet_path_list, size, transform=None, distort=False, overlap=None, mode="train") :
		'''
			path_list : list of path of alphabet_folders to train;
			size : size of dataset
		'''
		self.alphabet_path_list = alphabet_path_list
		self.size = size
		self.transform = transform
		self.distort = distort
		self.pairs = []
		self.labels = []
		# self.transforms = [] # what transform to do...

		if mode == "train" :
			for data_iter in range(0, self.size) :
				# randomly select alphabet; and its character
				alphabet_path = random.choice(alphabet_path_list)
				char_list = os.listdir(alphabet_path)

				char_path_1 = self.pop_subpath(char_list, alphabet_path)

				# choose to add whther same or different pair
				self.labels.append(random.getrandbits(1))
				is_same = bool(self.labels[-1])

				drawer_range = range(1, 13)

				image_list_1 = [p for p in os.listdir(char_path_1) if int(p[5:7]) in drawer_range]
				image_path_1 = self.pop_subpath(image_list_1, char_path_1)

				if is_same :
					char_path_2 = char_path_1
					image_list_2 = image_list_1
				else :
					char_path_2 = self.pop_subpath(char_list, alphabet_path)
					image_list_2 = [p for p in os.listdir(char_path_2) if int(p[5:7]) in drawer_range]

				image_path_2 = self.pop_subpath(image_list_2, char_path_2)

				self.pairs.append((image_path_1, image_path_2))

		elif mode == "valid" or mode == "test" :
			for alphabet_path in alphabet_path_list :
				drawer_range = range(1, 21)
				if overlap is not None and alphabet_path in overlap :
					drawer_range = range(13, 17) if mode == "valid" else range(17, 21)

				drawer_list = [idx for idx in drawer_range]
				# twice per alphabet
				for twice in range(0, 2) :
					char_list = os.listdir(alphabet_path)[0:20]
					char_list_tgt = char_list[:]

					drawer_1 = drawer_list.pop(random.randrange(len(drawer_list)))
					drawer_2 = drawer_list.pop(random.randrange(len(drawer_list)))

					for char in char_list :
						char_path = os.path.join(alphabet_path, char)
						image_name = [p for p in os.listdir(char_path) if int(p[5:7]) == drawer_1][0]
						image_path = os.path.join(char_path, image_name)

						for char_tgt in char_list_tgt :
							char_path_tgt = os.path.join(alphabet_path, char_tgt)
							image_name_tgt = [p for p in os.listdir(char_path_tgt) if int(p[5:7]) == drawer_2][0]
							image_path_tgt = os.path.join(char_path_tgt, image_name_tgt)

							self.pairs.append((image_path, image_path_tgt))
							if char == char_tgt :
								self.labels.append(1)
							else :
								self.labels.append(0)

				if len(self.pairs) >= self.size :
					assert(len(self.labels) == self.size)
					break


	def pop_subpath(self, listdir, base_path, pop_index=None) :
		if pop_index is None :
			pop_index = random.randrange(len(listdir))

		popped = listdir.pop(pop_index)

		return os.path.join(base_path, popped)

	def __getitem__(self, index) :
		image_path_1, image_path_2 = self.pairs[index]
		label = self.labels[index]

		image_1 = Image.open(image_path_1).convert('L')
		image_2 = Image.open(image_path_2).convert('L')

		if self.transform is not None :
			image_1 = self.transform(image_1)
			image_2 = self.transform(image_2)

		return image_1, image_2, label

	def __len__(self) :
		return len(self.pairs)


def get_loader(bg_path, ev_path, batch_size, dataset_size, shuffle, transform, mode="train") :
	alphabet_path_list = [os.path.join(bg_path, alphabet) for alphabet in os.listdir(bg_path)]

	if mode == "train" :
		siamese = SiameseDataset(alphabet_path_list, dataset_size, transform=transform, mode=mode)
	else :
		eval_path_list = sorted([os.path.join(ev_path, alphabet) for alphabet in os.listdir(ev_path)])
		if mode == "valid" :
			total_path_list = eval_path_list[0:10] + alphabet_path_list
		elif mode == "test" :
			total_path_list = eval_path_list[10:] # + alphabet_path_list

		siamese = SiameseDataset(total_path_list, dataset_size, transform=transform, overlap=alphabet_path_list, mode=mode)

	data_loader = DataLoader(dataset=siamese,
							batch_size=batch_size,
							shuffle=shuffle)

	return data_loader