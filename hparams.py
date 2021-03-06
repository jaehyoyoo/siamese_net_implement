import argparse

class Hparams :
	def __init__(self, args) :
		self.model_path = args.model_path
		self.bg_data_path = args.bg_data_path
		self.ev_data_path = args.ev_data_path
		self.log_step = args.log_step
		self.save_step = args.save_step

		self.num_epochs = args.num_epochs
		self.batch_size = args.batch_size
		self.dataset_size = args.dataset_size
		self.valid_trial = args.valid_trial
		self.test_trial = args.test_trial
		self.num_way = args.num_way
		self.learning_rate = args.learning_rate
		self.momentum = args.momentum
		self.reg_scale = args.reg_scale


def get_hparams() :
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='model/' , help='path for saving trained models')
	parser.add_argument('--bg_data_path', type=str, default="./data/images_background", help='path for background data')
	parser.add_argument('--ev_data_path', type=str, default="./data/images_evaluation", help='path for evaluation data')
	parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
	
	parser.add_argument('--num_epochs', type=int, default=5)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--dataset_size', type=int, default=30000)
	parser.add_argument('--valid_trial', type=int, default=320, help="must be multiplies of 2*num_way")
	parser.add_argument('--test_trial', type=int, default=400, help="must be multiplies of 2*num_way")
	parser.add_argument('--num_way', type=int, default=20)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	parser.add_argument('--momentum', type=float, default=0.5)
	parser.add_argument('--reg_scale', type=float, default=0.1)
	args = parser.parse_args()
	print(args)

	return Hparams(args)