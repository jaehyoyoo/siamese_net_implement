import torch
import torch.nn as nn
import time
import os
import numpy as np

from data_loader import get_loader
from hparams import get_hparams
from model import SiameseNet

from torchvision import transforms



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main() :
	hp = get_hparams()
	transform = transforms.Compose([
		transforms.ToTensor()])

	train_loader = get_loader(hp.bg_data_path, hp.ev_data_path,
			hp.batch_size, hp.dataset_size, True, transform, mode="train")
	valid_loader = get_loader(hp.bg_data_path, hp.ev_data_path,
			hp.num_way, hp.valid_trial * hp.num_way, False, transform, mode="valid")
	test_loader = get_loader(hp.bg_data_path, hp.ev_data_path,
			hp.num_way, hp.test_trial * hp.num_way, False, transform, mode="test")
	model = SiameseNet().to(device)

	def weights_init(m) :
		if isinstance(m, nn.Conv2d) :
			torch.nn.init.normal_(m.weight, 0.0, 1e-2)
			torch.nn.init.normal_(m.bias, 0.5, 1e-2)

		if isinstance(m, nn.Linear) :
			torch.nn.init.normal_(m.weight, 0.0, 0.2)
			torch.nn.init.normal_(m.bias, 0.5, 1e-2)

	model.apply(weights_init)
	
	num_epochs = hp.num_epochs
	total_step = len(train_loader)
	stop_decision = 1
	prev_error = 0.0

	for epoch in range(num_epochs) :
		lr = hp.learning_rate * pow(0.99, epoch)
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=hp.momentum, weight_decay=hp.reg_scale)

		for i, (images_1, images_2, label) in enumerate(train_loader) :
			images_1 = images_1.to(device).float()
			images_2 = images_2.to(device).float()
			label = label.to(device).float()

			prob = model(images_1, images_2)
			obj = label * torch.log(prob) + (1. - label) * torch.log(1. - prob)
			loss = -torch.sum(obj) / float(hp.batch_size)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i + 1) % hp.log_step == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
	                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

		valid_errors = 0.0
		total_sample = 0.0

		for images_1, images_2, label in valid_loader :
			images_1 = images_1.to(device).float()
			images_2 = images_2.to(device).float()
			label = label.to(device).float()
			prob = model(images_1, images_2)
			obj = label * torch.log(prob) + (1. - label) * torch.log(1. - prob)
			valid_errors += -torch.sum(obj).detach().cpu().numpy() / float(hp.num_way)
			total_sample += 1.0

		valid_error = np.round(valid_errors / total_sample, 4)

		print('Epoch [{}/{}], Validation Error : {:.4f}'
				.format(epoch+1, num_epochs, valid_error))

		if valid_error == prev_error :
			stop_decision += 1
		else :
			stop_decision = 1

		if stop_decision == 20 :
			print('Epoch [{}/{}], Early Stopped Training!'.format(epoch+1, num_epochs))
			torch.save(model.state_dict(), os.path.join(
				hp.model_path, 'siamese-{}.ckpt'.format(epoch+1)))
			break

		prev_error = valid_error

		if (epoch + 1) % 20 == 0 :
			torch.save(model.state_dict(), os.path.join(
				hp.model_path, 'siamese-{}.ckpt'.format(epoch+1)))

if __name__ == '__main__' :
	main()
