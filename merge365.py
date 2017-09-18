import cv2
import json

import os
root = '/home/zhou/place365'
img_list = 'places365_val.txt'
# img_data = 'test_large'
cat  = 'categories_places365.txt'
# f = open('/home/zhou/ai_challenger_scene_train_20170904/trainlist_merge.json','r')
merge = open('/home/zhou/ai_challenger_scene_train_20170904/train_merge.txt','a+')
f2 = open('/home/zhou/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json','r')
# data = json.load(f)
data = json.load(f2)

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




file = open(os.path.join(root,img_list)).readlines()
category = open(os.path.join(root,cat)).readlines()
new_dict = {v:k for k,v in table.items()}
def fetch(name,to_name,total):
    to_num = new_dict[to_name]
    for c in category:
        num = c.strip('\n').split(' ')[-1]
        cat = c.strip('\n').split('/')[2].split(' ')[0]
        if cat == name:
            counter = 0
            for i, nm in enumerate(file):
                path, n = nm.strip('\n').split(' ')

                if n == num:
                    dic = path+' '+str(to_num)+'\n'
                    print dic
                    counter+=1
                    merge.write(dic)
                    if counter>total:
                        break
            print counter
            break


 # {'001': 'a', '002': 'b'}
# counter = 0
# for line in data:
#     image = line['image_id']
#     label = line['label_id']
#     merge.write(image+' '+label+'\n')
#     counter+=1
#     print counter


# fetch('basketball_court','basketball_court',500)



