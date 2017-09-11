from packages import *
import json
import os
import random
import cv2
from os import listdir
import numpy as np
import torch.utils.data as data

import h5py
class Loader(data.Dataset):

    def __init__(self,root,list_file,train,transform,size ):
        self.root = root
        self.train = train
        self.transform = transform
        self.image_size = 448
        f = open(list_file, 'r')
        self.datas = json.load(f)
        self.num_samples = len(self.datas)
    def prepare_datas(self):
        name = 'scene_train_annotations_20170904.json'
        f = open(os.path.join('/home/zhou/ai_challenger_scene_validation_20170908'
, name))
        datas = json.load(f)
        num = len(datas)
        traindatas = []
        traininds = range(num)
        # testdatas = []
        # testinds = random.sample(range(num), num / 10)
        # for ind in testinds:
        #     testdatas.append(datas[ind])

        # traininds = list(set(testinds) ^ set(range(num)))
        for ind in traininds:
            traindatas.append(datas[ind])
        train = open('trainlist.json','w')
        # test = open('testlist.json','w')
        json.dump(traindatas,train)
        # json.dump(testdatas,test)
        train.close()
        # test.close()
        print 'prepare data successfully'
    def __getitem__(self,idx):
        data = self.datas[idx]
        image = cv2.imread(os.path.join(self.root,data['image_id']))
        label = data['label_id']
        image = cv2.resize(image,(self.image_size,self.image_size))
        image = self.transform(image)
        target = int(label)
        return image,target
    #
    # def get_avg_and_std(self):
    #
    #     image_path = 'scene_train_images_20170904'
    #     name = 'scene_train_annotations_20170904.json'
    #     f = open(os.path.join(self.path, name))
    #     datas = json.load(f)
    #     images = []
    #     # labels = []
    #     # images = np.empty((53878,448,448,3))
    #     avg = np.array([0.,0.,0.])
    #     std = np.array([0.,0.,0.])
    #     for i,data in enumerate(datas):
    #         image = cv2.imread(os.path.join(self.path, image_path, data['image_id']))
    #         image = cv2.resize(image,(448,448))/255.
    #         # labels.append(np.array([data['label_id']]).astype(np.int8)[np.newaxis,...])
    #         avg = np.average(np.average(image,1),0)*((1.)/(i+1.))+(i)/(i+1.)*avg
    #         std = np.std(np.std(image,1),0)*(1./(i+1.))+(i)/(i+1.)*std
    #
    #         # images[i][...] = image[...]
    #         print avg,std

        # labels = np.concatenate(labels,0)
        # file = h5py.File('TrainSet.h5','w')
        # file.create_dataset('images',data = images)
        # file.create_dataset('labels',data = labels)
    def __len__(self):
        return self.num_samples

class Indoor(data.Dataset):
    image_size =448
    def __init__(self,root,list_file,train,transform ):
        self.root = root
        self.train = train
        self.transform = transform
        self.map = {'studiomusic': 28, 'office': 23, 'hospitalroom': 35, 'operating_room': 59, 'children_room': 3, 'inside_subway': 25, 'library': 2, 'nursery': 42, 'elevator': 19, 'bedroom': 53, 'florist': 33, 'airport_inside': 12, 'livingroom': 22, 'closet': 43, 'artstudio': 15, 'tv_studio': 54, 'bathroom': 41, 'computerroom': 52, 'clothingstore': 0, 'videostore': 18, 'winecellar': 48, 'bakery': 51, 'restaurant_kitchen': 6, 'corridor': 21, 'warehouse': 24, 'grocerystore': 17, 'bookstore': 66, 'shoeshop': 4, 'kindergarden': 61, 'hairsalon': 10, 'fastfood_restaurant': 64, 'deli': 65, 'trainstation': 56, 'gym': 39, 'laundromat': 44, 'church_inside': 32, 'buffet': 1, 'meeting_room': 50, 'toystore': 29, 'laboratorywet': 62, 'pantry': 11, 'jewelleryshop': 34, 'locker_room': 46, 'auditorium': 47, 'greenhouse': 57, 'lobby': 45, 'kitchen': 26, 'classroom': 27, 'waitingroom': 14, 'bar': 58, 'restaurant': 49, 'dining_room': 38, 'casino': 31, 'stairscase': 40, 'inside_bus': 37, 'dentaloffice': 9, 'garage': 60, 'mall': 7, 'gameroom': 55, 'poolinside': 20, 'subway': 16, 'bowling': 5, 'cloister': 13, 'movietheater': 8, 'concert_hall': 30, 'museum': 63, 'prisoncell': 36}

        f = open(list_file, 'r')
        self.datas = f.readlines()
        self.num_samples = len(self.datas)

    def __getitem__(self,idx):
        data = self.datas[idx].strip('\n')
        image = cv2.imread(os.path.join(self.root,data))
        label = self.map[data.split('/')[0]]
        try:
            image = cv2.resize(image,(self.image_size,self.image_size))
        except:
            pass
        image = self.transform(image)
        target = int(label)
        return image,target
    def clear(self):
        f = open("TestImages.txt",'w')
        for idx in range(self.num_samples):
            data = self.datas[idx].strip('\n')
            image = cv2.imread(os.path.join(self.root, data))
            if image is None:
                print'xxxx'
            else:
                f.write(data+'\n')
                print"oooo"
    #
    # def get_avg_and_std(self):
    #
    #     image_path = 'scene_train_images_20170904'
    #     name = 'scene_train_annotations_20170904.json'
    #     f = open(os.path.join(self.path, name))
    #     datas = json.load(f)
    #     images = []
    #     # labels = []
    #     # images = np.empty((53878,448,448,3))
    #     avg = np.array([0.,0.,0.])
    #     std = np.array([0.,0.,0.])
    #     for i,data in enumerate(datas):
    #         image = cv2.imread(os.path.join(self.path, image_path, data['image_id']))
    #         image = cv2.resize(image,(448,448))/255.
    #         # labels.append(np.array([data['label_id']]).astype(np.int8)[np.newaxis,...])
    #         avg = np.average(np.average(image,1),0)*((1.)/(i+1.))+(i)/(i+1.)*avg
    #         std = np.std(np.std(image,1),0)*(1./(i+1.))+(i)/(i+1.)*std
    #
    #         # images[i][...] = image[...]
    #         print avg,std

        # labels = np.concatenate(labels,0)
        # file = h5py.File('TrainSet.h5','w')
        # file.create_dataset('images',data = images)
        # file.create_dataset('labels',data = labels)
    def __len__(self):
        return self.num_samples

