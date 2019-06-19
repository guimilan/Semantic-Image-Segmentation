import main
from original_fcn_alexnet import OriginalFCNAlexnet
import torch

def test():
	print('setting device...')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#device = torch.device("cpu")
	print('device set')

	print('creating original fcn')
	fcn = OriginalFCNAlexnet(num_classes=5)
	fcn.to(device)
	print('original fcn successfully created')

	print('creating coco data loader')
	data_loader = main.create_coco_data_loader(batch_size=8, shuffle=True)
	print('coco data loader successfully created')
	
	main.fit(model=fcn, train_dataset=data_loader, device=device, epoch=0, image_index=0, optimizer=None)
	return 0

if __name__ == '__main__':
	test()