import torch
import time

from data_loader import get_loader
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw_image(image) :
	for sample in image :
		for row in sample[0] :
			string = ""
			for ele in row :
				if ele == 1. :
					string += " "
				else :
					string += "O"

			print(string)


def main() :
	transform = transforms.Compose([
		transforms.Resize(30),
		transforms.ToTensor()])

	loader = get_loader("./data/images_background", 1, 10, True, transform)

	for i, (images_1, images_2, label) in enumerate(loader) :
		print("example %d" % i)
		draw_image(images_1.numpy())
		draw_image(images_2.numpy())
		
		label = label.numpy()
		if label == 1 : 
			print("same")
		else :
			print("different")

		images_1 = images_1.to(device)

		time.sleep(3)
		print("\n\n")

if __name__ == '__main__' :
	main()
