import cv2
import torch
from utils import Loader,testLoader
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from Network import densenet121
import torch.optim as optim
import json
from  augmentations import Augmentation
from inception_resnet import inceptionresnetv2
import os
import torch.backends.cudnn as cudnn
from Network import densenet121,FocalLoss,resnet50
import matplotlib.pyplot as plt
argument  =Augmentation()
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean= ( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])

data_path = '/home/zhou/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922'

names = os.listdir(data_path)


net = inceptionresnetv2(num_classes=1000)
net = torch.nn.DataParallel(net,device_ids=[0,1])

net.cuda()
net.eval()
cudnn.benchmark = True
net.load_state_dict(torch.load('supermerge/epoch_5_top1_0.810_top3_0.943.pkl'))

sumit = []

for num , name in enumerate(names) :

    label_id =[]
    file = data_path+'/'+name
    img = cv2.imread(file)
    img = cv2.resize(img,(299,299))
    images = [transform(img)]
    for i in range(9):
        ag_image = argument(img)
        images.append(ag_image)
    images = torch.stack(images, 0)
    images = Variable(images.cuda())
    out = F.softmax(net(images)[:, :80
                    ]).cpu().data
    out = out.sum(0).unsqueeze(0)
    for i in range(3):
        _, max = torch.max(out, 1)
        label_id.append(max[0])
        out[0][max[0]] = -1

    sumit.append({"image_id":name,"label_id":label_id})

    print num

json_file = open('submit.json','w')
json.dump(sumit,json_file)