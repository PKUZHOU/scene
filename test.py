import cv2
import torch
from utils import Loader
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from Network import densenet121
import torch.optim as optim
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
78	:	'rodeo',
79	:	'igloo/ice_engraving'}


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean= ( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
# trainset = Loader(root = '/home/zhou/ai_challenger_scene_train_20170904/scene_train_images_20170904',list_file='/home/zhou/ai_challenger_scene_train_20170904/trainlist.json',train = True,transform = transform)
testset = Loader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json',train=False,transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset,batch_size = 16,shuffle = True,num_workers = 4)
testloader = torch.utils.data.DataLoader(testset,batch_size = 8,shuffle = False,num_workers = 4)

net = densenet121(False)
net = torch.nn.DataParallel(net,device_ids=[0,1])
net.cuda()
net.eval()
cudnn.benchmark = True
net.load_state_dict(torch.load('dilation/epoch_35_top1_0.566_top3_0.759.pkl'))
counter = 0
correct = 0
top1 = 0
top3 = 0
counter = 0
for batch_id, (images, target) in enumerate(testloader):
    images = Variable(images.cuda())
    out = net(images).cpu().data
    size = out.size(0)
    _, max = torch.max(out,1)
    for i in range(size):
        label = target[i]
        pred = out[i][label]
        topk=0

        for cls in range(80):
            if out[i][cls] >pred:
                topk+=1
        if topk ==0:
            top1+=1
            top3+=1


        elif 0<topk<=2:
            top3+=1
        # print 'true:',table[label]
        # print 'pred:',table[max[i][0]]
        # image = images.data.cpu()[i]
        # image = image.permute(1, 2, 0).numpy()
        # image = (image * [0.225, 0.225, 0.225] + [0.44731586, 0.47744268, 0.49484214])
        # cv2.imshow('test', image)
        # cv2.waitKey()
    counter +=8

    print batch_id
print 'top3:', float(top3) / counter,'top1:',float(top1)/counter