import av
import glob
import os
import cv2
import numpy as np
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import sys
from model import resnet50


def extract_frames(video_path):
	video = av.open(video_path)
	frames = [frame.to_image() for frame in video.decode(0)]
	return frames

def vid_to_image(vid_path):
	seq_length = 16
	frame_list = extract_frames(vid_path)
	skip_length = int(len(frame_list)/seq_length)

	k = 0
	l = 0
	while(l!=seq_length):
		p3 = frame_list[k]
		open_cv_image = np.array(p3)
		img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
		img = cv2.resize(img,(128,128))
		if(k==0):
			img1 = img
		else:
			img1 = np.append(img1,img,axis = 1)
		k = k+skip_length
		l = l+1    
	return img1

def transform_image(index):
	im_size = 128
	mean = [0.4889, 0.4887, 0.4891]
	std = [0.2074, 0.2074, 0.2074]
	test_transforms = transforms.Compose([  transforms.ToPILImage(),
											transforms.Resize((im_size,im_size)),
											transforms.ToTensor()])
	test_data = video_dataset(index,sequence_length = 16,transform = test_transforms)
	test_loader = DataLoader(test_data,batch_size = 1,num_workers = 4 ,shuffle = False)
	dataloaders = {'test':test_loader}
	return dataloaders

class video_dataset(Dataset):
	def __init__(self,frame_list,sequence_length = 16,transform = None):
		self.frame_list = frame_list
		self.transform = transform
		self.sequence_length = sequence_length
	def __len__(self):
		return len(self.frame_list)
	def __getitem__(self,idx):
		im_size = 128
		label,path = self.frame_list[idx]  
		open_cv_image = np.array(path)
		img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)  
		seq_img = list()
		for i in range(16):
			img1 = img[:,128*i:128*(i+1),:]
			if(self.transform):
				img1 = self.transform(img1)
			seq_img.append(img1)
		seq_image = torch.stack(seq_img)
		seq_image = seq_image.reshape(3,16,im_size,im_size)
		return seq_image

def get_prediction(model,dataloaders):
	values = ['','Assult','Normal','Arrest','Explosion']
	for batch_i, X in enumerate(dataloaders['test']):
		image_sequences = Variable(X)
		predictions = model(image_sequences)
		return values[predictions.detach().argmax(1).item()]

def main(vid_path):
	model_path = 'c3d_48.h5'
	final_image = vid_to_image(vid_path)
	index = list()
	index.append(('vid',final_image))
	dataloaders = transform_image(index)
	model = resnet50(class_num=8)
	model.load_state_dict(torch.load(model_path))
	return get_prediction(model,dataloaders)

if __name__ == '__main__':
	vid_path = '../1.mp4'
	prediction = main(vid_path)
	print(prediction)