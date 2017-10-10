import cv2
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean= ( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
import torch
import random
from inception_resnet import inceptionresnetv2
# net = inceptionresnetv2(num_classes=1000)
import torch.backends.cudnn as cudnn
# from torch.autograd import Variable
# net.cuda()
# net.eval()
# cudnn.benchmark = True
# net = torch.nn.DataParallel(net,device_ids=[0,1])
# net.load_state_dict(torch.load('merge/epoch_11_top1_0.810_top3_0.942.pkl'))
map = {'landing_field':['airfield'],'airplane_cabin':['airplane_cabin'],'amusement_park':['amusement_park'],'skating_rink':['ice_skating_rink/indoor','ice_skating_rink/outdoor'],'arena/performance':[None],'art_room':['art_studio','art_school'],'assembly_line':['assembly_line'],'baseball_field':['baseball_field'],'football_field':['football_field'],'soccer_field':['soccer_field'],'volleyball_court':['volleyball_court/outdoor'],'golf_course':['golf_course'],'athletic_field':['athletic_field/outdoor'],'ski_slope':['ski_slope'],'basketball_court':['basketball_field'],'gymnasium':['gymnasium/indoor'],'bowling_alley':['bowling_alley'],'swimming_pool':['swimming_pool/outdoor','swimming_pool/indoor'],'boxing_ring':['boxing_ring'],'racecourse':['racecourse'],'arm/farm_field':['farm'],'orchard/vegetable':['orchard'],'pasture':['pasture'],'countryside':[None],'greenhouse':[None],'television_studio':['television_studio'],'temple/east_asia':['temple/asia'],'pavilion':['pavilion'],'tower':['tower'],'palace':['palace'],'church':['church/indoor','church/outdoor'],'street':['street'],'dining_room':['dining_hall','dining_room'],'coffee_shop':['coffee_shop'],'kitchen':['kitchen'],'plaza':['plaza'],'laboratory':[None],'bar':['bar'],'conference_room':['conference_room'],'office':['office'],'hospital':['hospital_room'],'ticket_booth':['ticket_booth'],'campsite':['campsite'],'music_studio':['music_studio'],'elevator/staircase':['elevator/staircase','elevator/staircase','staircase'],'arden':[None],'construction_site':['construction_site'],'general_store':['general_store/indoor'],'clothing_store':['clothing_store'],'azaar':[None],'library/bookstore':['library/indoor','bookstore'],'classroom':['classroom'],'ocean/beach':['ocean','beach'],'firefighting':[None],'gas_station':['gas_station'],'landfill':['landfill'],'balcony':['balcony/interior'],'recreation_room':['recreation_room'],'discotheque':['discotheque'],'museum':['museum/indoor'],'desert/sand':['desert/sand'],'raft':['raft'],'forest':['forest_path','forest/broadleaf'],'bridge':['bridge'],'residential_neighborhood':[None],'auto_showroom':['auto_showroom'],'lake/river':['lake/natural','river'],'aquarium':['aquarium'],'aqueduct':[None],'banquet_hall':['banquet_hall'],'bedchamber':['bedchamber'],'mountain':['mountain','mountain_path','mountain_snowy'],'station/platform':['subway_station/platform'],'lawn':['lawn'],'nursery':['nursery'],'beauty_salon':['beauty_salon'],'repair_shop':['repair_shop'],'rodeo':['arena/rodeo'],'igloo/ice_engraving':['igloo']}
# dismap = {}
# for k,v in map.items():
#     for x in v:
#         if not x == None:
#             dismap[x] = k
# print dismap
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
import os

# merge = open('/home/zhou/ai_challenger_scene_train_20170904/train_merge.txt','a+')
class shuf_labels:
    def __init__(self):
        self.cat_365 = open('/home/zhou/place365/categories_places365.txt','r').readlines()
        self.raw_list = open('/home/zhou/ai_challenger_scene_train_20170904/merge_temp/store.txt','r').readlines()
        self.list365 = open('/home/zhou/place365/places365_train_standard.txt','r').readlines()
        self.table = {'firefighting': 54, 'bedchamber': 71, 'airplane_cabin': 2, 'museum': 60, 'laboratory': 37, 'campsite': 43, 'amusement_park': 3, 'landing_field': 1, 'music_studio': 44, 'television_studio': 26, 'elevator/staircase': 45, 'street': 32, 'volleyball_court': 11, 'greenhouse': 25, 'repair_shop': 77, 'church': 31, 'pasture': 23, 'beauty_salon': 76, 'football_field': 9, 'gas_station': 55, 'mountain': 72, 'racecourse': 20, 'balcony': 57, 'basketball_court': 15, 'general_store': 48, 'aqueduct': 69, 'coffee_shop': 34, 'hospital': 41, 'athletic_field': 13, 'orchard/vegetable': 22, 'desert/sand': 61, 'lake/river': 67, 'forest': 63, 'construction_site': 47, 'auto_showroom': 66, 'raft': 62, 'kitchen': 35, 'ocean/beach': 53, 'bridge': 64, 'classroom': 52, 'arden': 46, 'tower': 29, 'library/bookstore': 51, 'igloo/ice_engraving': 79, 'countryside': 24, 'banquet_hall': 70, 'office': 40, 'golf_course': 12, 'ticket_booth': 42, 'gymnasium': 16, 'conference_room': 39, 'assembly_line': 7, 'aquarium': 68, 'palace': 30, 'bar': 38, 'skating_rink': 4, 'boxing_ring': 19, 'bowling_alley': 17, 'soccer_field': 10, 'recreation_room': 58, 'lawn': 74, 'arena/performance': 5, 'pavilion': 28, 'airport_terminal': 0, 'station/platform': 73, 'plaza': 36, 'arm/farm_field': 21, 'art_room': 6, 'landfill': 56, 'swimming_pool': 18, 'rodeo': 78, 'nursery': 75, 'discotheque': 59, 'temple/east_asia': 27, 'baseball_field': 8, 'clothing_store': 49, 'residential_neighborhood': 65, 'ski_slope': 14, 'dining_room': 33, 'azaar': 50}
        self.map = {'landing_field':['airfield'],'airplane_cabin':['airplane_cabin'],'amusement_park':['amusement_park'],'skating_rink':['ice_skating_rink/indoor','ice_skating_rink/outdoor'],'arena/performance':[None],'art_room':['art_studio','art_school'],'assembly_line':['assembly_line'],'baseball_field':['baseball_field'],'football_field':['football_field'],'soccer_field':['soccer_field'],'volleyball_court':['volleyball_court/outdoor'],'golf_course':['golf_course'],'athletic_field':['athletic_field/outdoor'],'ski_slope':['ski_slope'],'basketball_court':['basketball_court/indoor'],'gymnasium':['gymnasium/indoor'],'bowling_alley':['bowling_alley'],'swimming_pool':['swimming_pool/outdoor','swimming_pool/indoor'],'boxing_ring':['boxing_ring'],'racecourse':['racecourse'],'arm/farm_field':['farm'],'orchard/vegetable':['orchard'],'pasture':['pasture'],'countryside':[None],'greenhouse':[None],'television_studio':['television_studio'],'temple/east_asia':['temple/asia'],'pavilion':['pavilion'],'tower':['tower'],'palace':['palace'],'church':['church/indoor','church/outdoor'],'street':['street'],'dining_room':['dining_hall','dining_room'],'coffee_shop':['coffee_shop'],'kitchen':['kitchen'],'plaza':['plaza'],'laboratory':[None],'bar':['bar'],'conference_room':['conference_room'],'office':['office'],'hospital':['hospital_room'],'ticket_booth':['ticket_booth'],'campsite':['campsite'],'music_studio':['music_studio'],'elevator/staircase':['elevator/staircase','elevator/staircase','staircase'],'arden':[None],'construction_site':['construction_site'],'general_store':['general_store/indoor'],'clothing_store':['clothing_store'],'azaar':[None],'library/bookstore':['library/indoor','bookstore'],'classroom':['classroom'],'ocean/beach':['ocean','beach'],'firefighting':[None],'gas_station':['gas_station'],'landfill':['landfill'],'balcony':['balcony/interior'],'recreation_room':['recreation_room'],'discotheque':['discotheque'],'museum':['museum/indoor'],'desert/sand':['desert/sand'],'raft':['raft'],'forest':['forest_path','forest/broadleaf'],'bridge':['bridge'],'residential_neighborhood':[None],'auto_showroom':['auto_showroom'],'lake/river':['lake/natural','river'],'aquarium':['aquarium'],'aqueduct':[None],'banquet_hall':['banquet_hall'],'bedchamber':['bedchamber'],'mountain':['mountain','mountain_path','mountain_snowy'],'station/platform':['subway_station/platform'],'lawn':['lawn'],'nursery':['nursery'],'beauty_salon':['beauty_salon'],'repair_shop':['repair_shop'],'rodeo':['arena/rodeo'],'igloo/ice_engraving':['igloo']}
        self.distable =  {0:'airport_terminal',
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
    def prepare(self):

        #convert place365 annos to aichallenger annos ,which stores in convert_list.txt

        map_365 = {'forest_path': 'forest', 'bridge': 'bridge', 'airplane_cabin': 'airplane_cabin', 'campsite': 'campsite','amusement_park': 'amusement_park', 'gymnasium/indoor': 'gymnasium', 'television_studio': 'television_studio','ski_slope': 'ski_slope', 'elevator/staircase': 'elevator/staircase', 'street': 'street','banquet_hall': 'banquet_hall', 'museum/indoor': 'museum', 'pasture': 'pasture','beauty_salon': 'beauty_salon', 'library/indoor': 'library/bookstore', 'football_field': 'football_field','balcony/interior': 'balcony', 'mountain': 'mountain', 'music_studio': 'music_studio','boxing_ring': 'boxing_ring', 'church/indoor': 'church', 'office': 'office', 'temple/asia': 'temple/east_asia',
         'coffee_shop': 'coffee_shop', 'bedchamber': 'bedchamber', 'desert/sand': 'desert/sand',
         'airfield': 'landing_field', 'farm': 'arm/farm_field', 'baseball_field': 'baseball_field',
         'subway_station/platform': 'station/platform', 'general_store/indoor': 'general_store',
         'construction_site': 'construction_site', 'dining_hall': 'dining_room', 'raft': 'raft',
         'bowling_alley': 'bowling_alley', 'beach': 'ocean/beach', 'ice_skating_rink/outdoor': 'skating_rink',
         'art_studio': 'art_room', 'palace': 'palace', 'arena/rodeo': 'rodeo', 'soccer_field': 'soccer_field',
         'lake/natural': 'lake/river', 'gas_station': 'gas_station', 'staircase': 'elevator/staircase',
         'mountain_snowy': 'mountain', 'volleyball_court/outdoor': 'volleyball_court', 'golf_course': 'golf_course',
         'racecourse': 'racecourse', 'ticket_booth': 'ticket_booth', 'nursery': 'nursery',
         'bookstore': 'library/bookstore', 'assembly_line': 'assembly_line', 'aquarium': 'aquarium',
         'church/outdoor': 'church', 'bar': 'bar', 'repair_shop': 'repair_shop', 'kitchen': 'kitchen',
         'classroom': 'classroom', 'recreation_room': 'recreation_room', 'lawn': 'lawn',
         'basketball_court/indoor': 'basketball_court', 'mountain_path': 'mountain', 'orchard': 'orchard/vegetable',
         'hospital_room': 'hospital', 'plaza': 'plaza', 'athletic_field/outdoor': 'athletic_field',
         'forest/broadleaf': 'forest', 'ocean': 'ocean/beach', 'landfill': 'landfill',
         'ice_skating_rink/indoor': 'skating_rink', 'discotheque': 'discotheque', 'conference_room': 'conference_room',
         'art_school': 'art_room', 'swimming_pool/indoor': 'swimming_pool', 'auto_showroom': 'auto_showroom',
         'clothing_store': 'clothing_store', 'pavilion': 'pavilion', 'tower': 'tower', 'river': 'lake/river',
         'dining_room': 'dining_room', 'swimming_pool/outdoor': 'swimming_pool', 'igloo': 'igloo/ice_engraving'}

        converted_list = open('/home/zhou/place365/converted_list.txt','w+')
        for line in self.list365:
            path,anno = line.strip('\n').split(' ')
            name = path[3:].split('/0')[0]
            if name in map_365.keys():
                new_anno = self.table[map_365[name]]
                anno = path+' '+str(new_anno)+'\n'
                converted_list.write(anno)
                print anno
    def select(self,num):
        converted_list = open('/home/zhou/place365/converted_list.txt', 'r').readlines()
        raw_list = self.raw_list
        merged_list = open('/home/zhou/place365/merged_list.txt','w+')
        for i in range(80):
            for line in raw_list:
                path, anno = line.strip('\n').split(' ')
                if int(anno) == i:

                    merged_list.write(line)
                    print line
            cout = 0
            for line in converted_list:
                path, anno = line.strip('\n').split(' ')

                if int(anno) == i:
                    path = '/home/zhou/ai_challenger_scene_train_20170904/scene_train_images_20170904'+path
                    # img = cv2.imread(path)
                    # img = cv2.resize(img,(299,299))
                    # img = Variable(transform(img).unsqueeze(0).cuda())

                    # _,label = torch.max(net(img).cpu().data,1)
                    label = label[0][0]
                    if label == i:
                        merged_list.write(line)
                        print line
                        cout +=1
                    else:
                        pass
                    if cout == num:
                        break
    def get_list(self):
        merged_list = open('/home/zhou/place365/merged_list.txt', 'r').readlines()
        temp_list = open('/home/zhou/place365/temp_list.txt', 'w+')
        temp_dict = {v:[] for v in range(80)}
        for line in merged_list:
            path, anno = line.strip('\n').split(' ')
            temp_dict[int(anno)]+=[line]
        maxnum =0
        for x in temp_dict.values():
            lenth = len(x)
            if lenth>maxnum:
                maxnum = lenth
        for x in temp_dict.values():
            y = random.sample(x,len(x))
            lenth = len(x)
            index = range(maxnum)
            for ind in index:
                img = y[ind%lenth]
                temp_list.write(img)
    def mask_select(self):
        merged_list = open('/home/zhou/place365/merged_list.txt', 'r').readlines()
        mask_temp_list = open('/home/zhou/place365/mask_temp_list.txt', 'w+')
# mask = ['museum','arena/performance','balcony','music_studio','arden','discotheque','coffee_shop','rodeo','aqueduct','lake/river','conference_room','nursery','library/bookstore','lawn','palace','baseball_field','beauty_salon','igloo/ice_engraving','bowling_alley','airplane_cabin','landing_field','banquet_hall','racecourse','tower']
        mask = ['museum','arena/performance','balcony','music_studio','arden','discotheque','coffee_shop','rodeo','aqueduct','lake/river','conference_room']



        temp_dict = {v: [] for v in range(80)}
        for line in merged_list:
            path, anno = line.strip('\n').split(' ')
            if self.distable[int(anno)] in mask:
                temp_dict[int(anno)] += [line]
        for line in self.raw_list:
            path, anno = line.strip('\n').split(' ')
            if not self.distable[int(anno)] in mask:
                temp_dict[int(anno)] += [line]

        maxnum = 0
        for x in temp_dict.values():
            lenth = len(x)
            if lenth > maxnum:
                maxnum = lenth
#   for x in temp_dict.values():
#           y = random.sample(x, len(x))
#           lenth = len(x)
#           index = range(maxnum)
#           for ind in index:
#               img = y[ind % lenth]
#               mask_temp_list.write(img)
		for x in temp_dict.values():
			for img in x:
#	print img:
				mask_temp_list.write(img)











# new_dict = {v:k for k,v in table.items()}
# print new_dict
#


# fetch('basketball_court','basketball_court',500)
# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean= ( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
#
# testset = testLoader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json',train=False,transform=transform,size=299)
# testloader = torch.utils.data.DataLoader(testset,batch_size = 1,shuffle = False,num_workers = 1)


# for batch_id, (images, target) in enumerate(testloader):
#
#     if table[target[0]] == 'igloo/ice_engraving':
#         image = images[0].permute(1, 2, 0).numpy()
#         image = (image * [0.225, 0.225, 0.225] + [0.44731586, 0.47744268, 0.49484214])
#         cv2.imshow('test', image)
#         cv2.waitKey()


# shuffle = shuf_labels()
# shuffle.get_list()
