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
from operator import itemgetter
from PIL import Image


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
	prediction = []
	prediction_lists = []
	for i in range(5):
		for batch_i, X in enumerate(dataloaders['test']):
			image_sequences = Variable(X)
			predictions = model(image_sequences)
			predictions = predictions.detach()
			list_x = predictions.tolist()[0]
			min_x = min(list_x)
			list_x2 = [i-min_x for i in list_x]
			sum_x = sum(list_x2)
			list_x3 = [i/sum_x for i in list_x2]
			prediction_lists.append(list_x3)
			prediction.append(predictions.argmax(1).item())
	final_prediction = max(set(prediction), key = prediction.count)
	lx = []
	k = -1
	for i in prediction:
		k += 1
		if i == final_prediction:
			if len(lx)>0:
				if lx[i] < prediction_lists[k][i]:
					lx = prediction_lists[k]
			else:
				lx = prediction_lists[k]
	final_list = [0 for i in range(5)]
	for i in range(4):
		final_list[i] = int(lx[i+1]*100)
	xl = sum(final_list)
	final_list[4] = 100-xl
	keyList = sorted(enumerate(final_list), key = itemgetter(1))
	for i in range(5):
		v = keyList[i][1]*((i+1)**3)
		keyList[i] = (keyList[i][0],v)
	theList = sorted(keyList, key = itemgetter(0))
	result = [record for key, record in theList]
	sux = sum(result)
	list_final = [int((i/sux)*100) for i in result]
	return values[final_prediction],list_final

def main(vid_path):
	model_path = 'c3d_48.h5'
	final_image = vid_to_image(vid_path)
	index = list()
	index.append(('vid',final_image))
	dataloaders = transform_image(index)
	model = resnet50(class_num=8)
	model.load_state_dict(torch.load(model_path))
	file_name = 'output_img.jpg'
	cv2.imwrite(file_name, final_image[:,128*8:128*9,:])
	return get_prediction(model,dataloaders)

if __name__ == '__main__':
	vid_path = '../1.mp4'
	prediction = main(vid_path)
	print(prediction)