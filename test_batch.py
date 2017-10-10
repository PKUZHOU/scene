import cv2
import torch
from utils import Loader,testLoader
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Network import densenet121
import torch.optim as optim
from inception_resnet import inceptionresnetv2

import torch.backends.cudnn as cudnn
from Network import densenet121,FocalLoss,resnet50
import matplotlib.pyplot as plt
table = {0:'airport_terminal',
1:'landing_field',
2	:	'airplane_cabin',
3	:'amusement_park',
4	:'skating_rink',
5	:'arena/performance',
6	:	'art_room',
7	:'assembly_line',
8	:'baseball_field',
9	:	'football_field',
10	:'soccer_field',
11	:'volleyball_court',
12	:	'golf_course',
13	:	'athletic_field',
14	:	'ski_slope',
15	:	'basketball_court',
16	:	'gymnasium',
17	:	'bowling_alley',
18	:	'swimming_pool',
19	:	'boxing_ring',
20	:	'racecourse',
21	:	'arm/farm_field',
22	:	'orchard/vegetable',
23	:	'pasture',
24	:	'countryside',
25	:	'greenhouse',
26	:	'television_studio',
27	:	'temple/east_asia',
28	:	'pavilion',
29	:	'tower',
30	:	'palace',
31	:	'church',
32	:	'street',
33	:	'dining_room',
34	:	'coffee_shop',
35	:	'kitchen',
36	:	'plaza',
37	:	'laboratory',
38	:	'bar',
39	:	'conference_room',
40	:	'office',
41	:	'hospital',
42	:   'ticket_booth',
43	:	'campsite',
44	:	'music_studio',
45	:	'elevator/staircase',
46	:	'arden',
47	:	'construction_site',
48	:	'general_store',
49	:	'clothing_store',
50	:	'azaar',
51	:	'library/bookstore',
52	:	'classroom',
53	:	'ocean/beach',
54	:	'firefighting',
55	:	'gas_station',
56	:	'landfill',
57	:	'balcony',
58	:	'recreation_room',
59	:	'discotheque',
60	:	'museum',
61	:	'desert/sand',
62	:	'raft',
63	:	'forest',
64	:	'bridge',
65	:	'residential_neighborhood',
66	:	'auto_showroom',
67	:	'lake/river',
68	:	'aquarium',
69	:	'aqueduct',
70	:	'banquet_hall',
71	:	'bedchamber',
72	:	'mountain',
73	:	'station/platform',
74	:	'lawn',
75	:	'nursery',
76	:	'beauty_salon',
77	:	'repair_shop',
78	:	'rodeo',
79	:	'igloo/ice_engraving'}

dic1 = dict(zip(table.values(),[0]*80))
dic2 = dict(zip(table.values(),[0]*80))
dic3 = dict(zip(table.values(),[0]*80))
# train_samples = {'firefighting': '0.99', 'bedchamber': '0.98', 'airplane_cabin': '1.0', 'museum': '0.99', 'laboratory': '0.97', 'campsite': '0.99', 'amusement_park': '0.99', 'landing_field': '0.99', 'music_studio': '0.99', 'television_studio': '0.97', 'elevator/staircase': '0.98', 'street': '0.96', 'volleyball_court': '0.99', 'greenhouse': '0.94', 'repair_shop': '0.99', 'church': '0.93', 'pasture': '0.94', 'beauty_salon': '0.99', 'football_field': '0.99', 'gas_station': '0.99', 'mountain': '0.97', 'racecourse': '0.99', 'balcony': '0.98', 'basketball_court': '1.0', 'general_store': '0.95', 'aqueduct': '0.98', 'coffee_shop': '0.87', 'hospital': '0.98', 'athletic_field': '0.99', 'orchard/vegetable': '0.94', 'desert/sand': '0.99', 'lake/river': '0.97', 'forest': '0.95', 'construction_site': '0.99', 'auto_showroom': '0.96', 'raft': '0.99', 'kitchen': '0.98', 'ocean/beach': '0.98', 'bridge': '0.99', 'classroom': '0.98', 'arden': '0.94', 'tower': '0.98', 'library/bookstore': '0.91', 'igloo/ice_engraving': '0.96', 'countryside': '0.91', 'banquet_hall': '0.98', 'office': '0.97', 'golf_course': '0.99', 'ticket_booth': '0.97', 'gymnasium': '0.99', 'conference_room': '0.98', 'assembly_line': '0.97', 'aquarium': '0.99', 'palace': '0.93', 'bar': '0.98', 'skating_rink': '0.99', 'boxing_ring': '0.99', 'bowling_alley': '1.0', 'soccer_field': '0.99', 'recreation_room': '0.98', 'lawn': '0.96', 'arena/performance': '0.98', 'pavilion': '0.95', 'airport_terminal': '0.98', 'station/platform': '0.99', 'plaza': '0.97', 'arm/farm_field': '0.95', 'art_room': '0.98', 'landfill': '0.99', 'swimming_pool': '0.99', 'rodeo': '0.99', 'nursery': '0.99', 'discotheque': '0.98', 'temple/east_asia': '0.94', 'baseball_field': '0.99', 'clothing_store': '0.96', 'residential_neighborhood': '0.87', 'ski_slope': '0.99', 'dining_room': '0.79', 'azaar': '0.98'}
# train_samples = {'firefighting': '0.84', 'bedchamber': '0.80', 'airplane_cabin': '0.94', 'museum': '0.65', 'laboratory': '0.84', 'campsite': '0.85', 'amusement_park': '0.80', 'landing_field': '0.92', 'music_studio': '0.81', 'television_studio': '0.63', 'elevator/staircase': '0.77', 'street': '0.68', 'volleyball_court': '0.96', 'greenhouse': '0.94', 'repair_shop': '0.67', 'church': '0.87', 'pasture': '0.6', 'beauty_salon': '0.84', 'football_field': '0.89', 'gas_station': '0.72', 'mountain': '0.70', 'racecourse': '0.93', 'balcony': '0.66', 'basketball_court': '0.98', 'general_store': '0.83', 'aqueduct': '0.81', 'coffee_shop': '0.46', 'hospital': '0.77', 'athletic_field': '0.90', 'orchard/vegetable': '0.60', 'desert/sand': '0.91', 'lake/river': '0.58', 'forest': '0.68', 'construction_site': '0.73', 'auto_showroom': '0.98', 'raft': '0.91', 'kitchen': '0.79', 'ocean/beach': '0.87', 'bridge': '0.75', 'classroom': '0.85', 'arden': '0.43', 'tower': '0.83', 'library/bookstore': '0.78', 'igloo/ice_engraving': '0.93', 'countryside': '0.40', 'banquet_hall': '0.80', 'office': '0.67', 'golf_course': '0.94', 'ticket_booth': '0.89', 'gymnasium': '0.94', 'conference_room': '0.79', 'assembly_line': '0.82', 'aquarium': '0.87', 'palace': '0.72', 'bar': '0.91', 'skating_rink': '0.92', 'boxing_ring': '0.96', 'bowling_alley': '0.91', 'soccer_field': '0.91', 'recreation_room': '0.75', 'lawn': '0.7', 'arena/performance': '0.72', 'pavilion': '0.83', 'airport_terminal': '0.9', 'station/platform': '0.84', 'plaza': '0.75', 'arm/farm_field': '0.75', 'art_room': '0.69', 'landfill': '0.88', 'swimming_pool': '0.90', 'rodeo': '0.79', 'nursery': '0.83', 'discotheque': '0.82', 'temple/east_asia': '0.73', 'baseball_field': '0.86', 'clothing_store': '0.64', 'residential_neighborhood': '0.46', 'ski_slope': '0.90', 'dining_room': '0.50', 'azaar': '0.81'}
#
# hs = []
#
# for x in range(80):
#     hs+= int(float(train_samples[table[x]])*100)*[x]
#
# plt.show(plt.hist(x = hs,bins = 80))
#


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean= ( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
trainset = Loader(root = '/home/zhou/ai_challenger_scene_train_20170904/scene_train_images_20170904',list_file='/home/zhou/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json',train = True,transform = transform,size= 299)
testset = testLoader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='en_trainlist.json',train=False,transform=transform,size=299)
trainset = Loader(root = '/home/zhou/ai_challenger_scene_train_20170904/scene_train_images_20170904',list_file='/home/zhou/ai_challenger_scene_train_20170904/merge_temp/store.txt',train = False,transform = transform,size=299)

# trainset = Loader(root = '/home/zhou/ai_challenger_scene_train_20170904/scene_train_images_20170904',list_file='/home/zhou/ai_challenger_scene_train_20170904/trainlist.json',train = True,transform = transform)
# testset = testLoader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json',train=False,transform=transform,size=299)
# trainloader = torch.utils.data.DataLoader(trainset,batch_size = 16,shuffle = True,num_workers = 4)
testloader = torch.utils.data.DataLoader(trainset,batch_size = 32,shuffle = False,num_workers = 4)
# net = torchvision.models.resnet101(pretrained=False)
# net = resnet50(pretrained=False,num_classes=80)
net = inceptionresnetv2(num_classes=1000)
net.classif = nn.Linear(1536, 80)

net = torch.nn.DataParallel(net,device_ids=[0,1])

net.cuda()
net.eval()
cudnn.benchmark = True
net.load_state_dict(torch.load('soft_label/epoch_7_top1_0.820_top3_0.944.pkl'))
counter = 0
correct = 0
top1 = 0
top3 = 0
counter = 0
for batch_id, (images, target) in enumerate(testloader):
    images = Variable(images.cuda())
    out = net(images).cpu().data[:,:80]
    size = out.size(0)
    _, max = torch.max(out,1)
    for i in range(size):
        label = target[i]
        pred = out[i][label]
        dic2[table[label]] += 1
        topk=0

        for cls in range(80):
            if out[i][cls] >pred:
                topk+=1
        if topk ==0:
            top1+=1

            top3+=1
            dic1[table[label]]+=1


        elif 0<topk\
                <=2:
            top3+=1
            # dic1[table[label]] += 1
        # print 'true:',table[label]
        # print 'pred:',table[max[i][0]]
        # # if table[label] == 'lake/river':
        # image = images.data.cpu()[i]
        # image = image.permute(1, 2, 0).numpy()
        # image = (image * [0.225, 0.225, 0.225] + [0.44731586, 0.47744268, 0.49484214])
        # cv2.imshow('test', image)
        # cv2.waitKey()
    counter +=32
    print batch_id
print 'top3:', float(top3) / counter,'top1:',float(top1)/counter
print dic1
print dic2
for i in range(80):
    dic3[table[i]] = str(dic1[table[i]]/float(dic2[table[i]]))[:4]

print dic3