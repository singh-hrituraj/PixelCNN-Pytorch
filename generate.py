'''
Code by Hrituraj Singh
Indian Institute of Technology Roorkee
'''

import sys
import os
import time
import torch
import torch.nn.functional as F

from utils import *
from Model import PixelCNN

def main(config_file):
	config = parse_config(config_file)
	model = config['model']
	images = config['images']

	save_path = model.get('save_path', '/home/gnpillai/Hrituraj/Pixel-CNN/Models')
	assert(os.path.exists(save_path), 'Saved Model File Does not exist!')
	no_images = images.get(no_images, 10)
	images_size = images.get(images_size, 28)
	images_channels images.get(images_channels, 1)

	#Define and load model
	net = PixelCNN()
	net.load_state_dict(torch.load(save_path))
	net.eval()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = PixelCNN().to(device)
	if torch.cuda.device_count() > 1:
  		print("Let's use", torch.cuda.device_count(), "GPUs!")
  		net = nn.DataParallel(net)


  	sample = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
  	sample.fill_(0)

  	#Generating images pixel by pixel
  	for i in range(images_size):
  		for j in range(images_size):
  			out = net(sample)
  			probs = F.softmax(out[:,:,i,j]).data()
  			sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.

  	#Saving images row wise
  	torch.utils.save_image(sample, 'sample.png', nrow=12, padding=0)





if __name__=='__main__':
	config_file = sys.argv[1]
	assert os.path.exists(config_file), "Configuration file does not exit!"
	main(config_file)