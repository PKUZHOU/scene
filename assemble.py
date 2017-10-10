import cv2
import torch
from utils import Loader,testLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from inception_resnet import inceptionresnetv2
from Network import ensemble_net,FocalLoss
import cPickle
import numpy as np
import random
import torch.backends.cudnn as cudnn
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
78	:	'rodeo',79	:	'igloo/ice_engraving'}


model_dic = {1:'supermerge/epoch_5_top1_0.810_top3_0.943.pkl',2:'supermerge/epoch_5_top1_0.806_top3_0.938.pkl',3:'merge/epoch_11_top1_0.810_top3_0.942.pkl'}

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean= ( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
testset = testLoader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='en_testlist.json',train=False,transform=transform,size=299)
testloader = torch.utils.data.DataLoader(testset,batch_size = 1,shuffle = False,num_workers = 1)
# net = inceptionresnetv2(num_classes=1000,pretrained=None)
# net.cuda()
# net.eval()
#
# net = torch.nn.DataParallel(net,device_ids=[0,1])
# data ={1:[],2:[],3:[],'target':[]}
# for k,v in model_dic.iteritems():
#     net.load_state_dict(torch.load(v))
#     print 'load '+v+'success'
#     for batch_id, (images, target) in enumerate(testloader):
#         images = Variable(images.cuda())
#         out = net(images)[:,:80]
#         out = F.softmax(out).cpu().data.numpy()
#         data[k].append([out])
#         data['target'].append(target[0])
#
#
#         print batch_id
#
# pkg_file = open('test.pkl', 'w')
# cPickle.dump(data, pkg_file)
train_file = open('train.pkl', 'r')
test_file = open('test.pkl','r')
#
train_data = cPickle.load(train_file)
test_data = cPickle.load(test_file)
train_file.close()
test_file.close()

def build_data(data):
    target = torch.from_numpy(np.asarray(data['target']))
    data1 = torch.from_numpy(np.asarray(data[1])).squeeze(1).squeeze(1)
    data2 = torch.from_numpy(np.asarray(data[2])).squeeze(1).squeeze(1)
    data3 = torch.from_numpy(np.asarray(data[3])).squeeze(1).squeeze(1)
    data  = torch.stack((data1,data2,data3),1)
    data = data.view(data.size(0),-1)
    target = target[:data.size(0)]
    return data,target
#
#
def get_batch(data,target,mask,batch_size):
    input_data = torch.zeros((batch_size,240))
    input_target = torch.zeros((batch_size,))
    for i,x in enumerate(mask):
        input_data[i][:] = data[x][:]
        input_target[i] = target[x]

    return input_data,input_target.long()
#
#
focal_loss = FocalLoss(gamma=2)
train_data,train_target = build_data(data=train_data)
test_data,test_target = build_data(data = test_data)

steps = [10,30,50]
batch_size = 16
lr = 0.01
momentum = 0.9
weight_decay = 0.0001
net = ensemble_net()
optimizer = optim.SGD(net.parameters(),lr = lr,momentum=momentum,weight_decay=weight_decay )
dic1 = dict(zip(table.values(),[0]*80))
dic2 = dict(zip(table.values(),[0]*80))
dic3 = dict(zip(table.values(),[0]*80))
for epoch in range(20000):
    epoch_loss = 0

    list = random.sample(range(train_data.size(0)),train_data.size(0))
    total_batch = len(list)/batch_size
    for batch_idx in range(total_batch):
        mask = list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        input_data,input_target = get_batch(train_data,train_target,mask,batch_size)
        true = input_data.view(input_data.size(0),3,80).sum(1).squeeze(1)
        _, max = torch.max(true, 1)
        input_data = Variable(input_data)
        input_target = Variable(input_target)
        input_data = input_data.view(input_data.size(0),3,80).permute(0,2,1)

        for i in range(input_data.size(0)):
            a = input_data[i]
            b =

        out = net(input_data)
        out = F.softmax(out)

        loss = focal_loss(out,input_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        epoch_loss = epoch_loss * (batch_idx / (batch_idx + 1.0)) + loss.data[0] * (1.0 / (batch_idx + 1.0))
    print "epoch_loss: %.4f"%(epoch_loss)
    if epoch%10 == 0:
        net.eval()
        top1 = 0
        top3 = 0
        counter = 0
        list = range(test_data.size(0))
        for i in list:
            input = Variable(test_data[i][:].unsqueeze(0))
            target = [test_target[i]]
            # true = input.data.view(input.size(0), 3, 80).sum(1).squeeze(1)/3.
            # _, max = torch.max(true, 1)
            out = net(input)
            out = F.softmax(out)
            # out = true
            # _, max = torch.max(out, 1)
            label = target[0]
            pred = out[0][label]
            dic2[table[label]] += 1
            topk = 0

            for cls in range(80):
                if out[0][cls] > pred:
                    topk += 1
            if topk == 0:
                top1 += 1

                top3 += 1
                dic1[table[label]] += 1
            elif 0 < topk <= 2:
                top3 += 1
                dic1[table[label]] += 1
            counter += 1
        print 'top3:', float(top3) / counter, 'top1:', float(top1) / counter
        print dic1
        print dic2
        for i in range(80):
            dic3[table[i]] = str(dic1[table[i]] / float(dic2[table[i]]))[:4]

        print dic3
    if epoch in steps:
        lr*=0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)



