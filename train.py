import cv2
import torch
from utils import Loader,Indoor,testLoader
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Network import densenet121,FocalLoss,resnet50,soft_label_Loss
from inception_resnet import inceptionresnetv2
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter
from augmentations import Augmentation

transform1 = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(),transforms.RandomCrop((299,299)),transforms.ToTensor(),transforms.Normalize(mean= ( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
# trainset = Loader(root = '/home/zhou/ai_challenger_scene_train_20170904/scene_train_images_20170904',list_file='/home/zhou/ai_challenger_scene_train_20170904/merge_temp/store.txt',train = True,transform = Augmentation(),size=299)
# testset = testLoader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json',train=False,transform=transform2,size=299)
trainset = testLoader(root='/home/zhou/ai_challenger_scene_train_20170904/scene_train_images_20170904',list_file='trainlist_two.json',train= True,transform= Augmentation(),size=299)
testset = testLoader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='en_testlist.json',train=False,transform=transform2,size=299)
# trainset = testLoader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='en_trainlist.json',train=False,transform=transform2,size=299)

trainloader = torch.utils.data.DataLoader(trainset,batch_size = 16,shuffle = True,num_workers = 4)
testloader = torch.utils.data.DataLoader(testset,batch_size = 16,shuffle = False,num_workers = 4)


##########################
#hyper parameters
lr = 0.01
weight_decay = 5e-4
start_epoch = 0
max_epoch = 30
resume = False
momentum = 0.9
forced_load = False
Freeze = False
show_images = False
##########################

net  = inceptionresnetv2(num_classes=1000,pretrained='imagenet')
net.classif = nn.Linear(1536, 80)

################################################################
################# Freeze some layers or no######################

lr_dic = []
if not Freeze:
    lr_dic = net.parameters()
else:
    free_layers = [net.classif, net.conv2d_7b, net.block8, net.mixed_7a, net.repeat_2, net.repeat_1, net.mixed_6a,
                   net.repeat, net.mixed_5b]
    for param in net.parameters():
        param.requires_grad = False
    for fl in free_layers:
        lr_dic.append({'params': fl.parameters()})
        for param in fl.parameters():
            param.requires_grad = True

################ load data complusively#########################
#
# if forced_load:
#     net = resnet50(pretrained=False,num_classes=80)
#     net = torchvision.models.resnet101(pretrained=True)
#     state_dict =torch.load('inceptionresnetv2-d579a627.pth')
#     own_state = net.state_dict()
#
#
#     for name, param in state_dict.items():
#         print name
#         if name not in own_state:
#             print "not"+name
#             pass
#         if isinstance(param, Parameter):
#             # backwards compatibility for serialized parameters
#             param = param.data
#         try:
#             own_state[name].copy_(param)
#             missing = set(own_state.keys()) - set(state_dict.keys())
#         except:
#             print 'unable to load'+name
#########
########################################################
net = torch.nn.DataParallel(net,device_ids=[0])
net.cuda()
#######################################################

if resume:
    net.load_state_dict(torch.load('soft_label/epoch_4_top1_0.819_top3_0.946.pkl'))

##############################################################
cudnn.benchmark = True
focalloss = FocalLoss(alpha=None,gamma=0.5,size_average=True)
soft_label_Loss =soft_label_Loss(e=0.01)
#steps when lr decays
steps = [3,7,12]
################################################################
optimizer = optim.SGD(lr_dic,lr = lr,momentum=momentum,weight_decay=weight_decay )

for epoch in range(start_epoch,max_epoch):
    ########################################
    #  lr decay
    if epoch in steps:
        lr *= 0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    print 'epoch:',epoch
    epoch_loss = 0
    net.train()
    for batch_idx,(images,target) in enumerate(trainloader):
        if show_images:
        ###################################################
        ##  show images
            for i in range(64):
                image = images[i]
                image = image.permute(1, 2, 0).numpy()
                image = (image * [0.225, 0.225, 0.225] + [0.44731586, 0.47744268, 0.49484214])
                cv2.imshow('test', image)
                cv2.waitKey()
        ##################################################
        images = Variable(images.cuda())
        target = Variable(target.cuda())
        out = net(images)
#        loss = F.cross_entropy(out,target)

        # loss = focalloss(out,target)
        loss = soft_label_Loss(out,target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_loss = epoch_loss*(batch_idx/(batch_idx+1.0))+loss.data[0]*(1.0/(batch_idx+1.0))
        if batch_idx%50 == 0:
            print 'batch:',batch_idx,'batch_loss: %.4f'%(loss.data[0]),"epoch_loss: %.4f"%(epoch_loss)

    #test

    #########################################################
    #test net
    print 'testing: epoch ',epoch
    counter = 0
    top3 = 0
    top1 = 0
    net.eval()
    for batch_id, (images, target) in enumerate(testloader):
        images = Variable(images.cuda())
        out = net(images).cpu().data
        size = out.size(0)

        for i in range(size):
            label = target[i]
            pred = out[i][label]
            topk = 0
            for cls in range(80):
                if out[i][cls] > pred:
                    topk += 1
            if topk == 0:
                top1 += 1
                top3 += 1
            elif 0 < topk <= 2:
                top3 += 1
        counter +=32
##############################################################
    print batch_id
    torch.save(net.state_dict(),'join/'+'epoch_'+repr(epoch)+'_top1_'+str(float(top1)/counter)[:5]+'_top3_'+str(float(top3)/counter)[:5]+'.pkl')
    print 'Top3 accuracy:' ,float(top3)/counter,' Top1 accuracy:' ,float(top1)/counter


