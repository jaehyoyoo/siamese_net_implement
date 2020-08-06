import torch
import time

from data_loader import get_loader
from hparams import get_hparams
from model import SiameseNet

from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main() :
	hp = get_hparams()

	transform = transforms.Compose([
		transforms.ToTensor()])

	train_loader = get_loader(hp.bg_data_path, hp.batch_size, hp.dataset_size, True, transform)
	model = SiameseNet().to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate, momentum=hp.momentum)

	num_epochs = hp.num_epochs
	total_step = len(train_loader)

	for epoch in range(num_epochs) :
		for i, (images_1, images_2, label) in enumerate(train_loader) :
			images_1 = images_1.to(device)
			images_2 = images_2.to(device)
			label = label.to(device)

			prob = model(images_1, images_2)
			loss = label * torch.log(prob) + (1. - label) * torch.log(1. - prob)
			loss = torch.sum(loss) / float(hp.batch_size)

			l1_norm = 0.0
			for param in model.parameters() :
				l1_norm += torch.norm(param.data, p=1).to(device)

			

			loss += hp.reg_scale * l1_norm

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i + 1) % hp.log_step == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
	                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

		torch.save(model.state_dict(), os.path.join(
			args.model_path, 'siamese-{}.ckpt'.format(epoch+1)))



if __name__ == '__main__' :
	main()
