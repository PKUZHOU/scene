import cv2
import torch
from utils import Loader,Indoor,ImgNet
import torchvision.transforms as transforms
import torchvision
from inception_resnet import inceptionresnetv2

import torch.nn.functional as F
from torch.autograd import Variable
from Network import densenet121,FocalLoss,resnet50
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter
from augmentations import Augmentation
import torch.backends.cudnn as cudnn
transform1 = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(),transforms.RandomCrop((299,299)),transforms.ToTensor(),transforms.Normalize(mean= ( 0.485 ,0.456,  0.406),std = ( 0.229 , 0.224 , 0.225))])


transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
trainset = ImgNet(root = '/home/zhou/data_256',list_file='/home/zhou/place365/places365_train_standard.txt',train = True,transform = transform1)
# testset = Loader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json',train=False,transform=transform2,size=224)
# trainset = Indoor(root = '/home/zhou/Images',list_file = 'TrainImages.txt',train= True,transform=transform)
# testset = Indoor(root = '/home/zhou/Images',list_file = 'TestImages.txt',train= False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size = 16,shuffle = True,num_workers = 8)
# testloader = torch.utils.data.DataLoader(testset,batch_size = 32,shuffle = False,num_workers = 8)


#hyper parameters

lr = 0.001
weight_decay = 5e-4
start_epoch = 0
max_epoch = 90
resume = False
momentum = 0.9
# net  = densenet121(pretrained=True)
# net = resnet50(pretrained=False,num_classes=80)
net = inceptionresnetv2(num_classes=1000,pretrained='imagenet')
net.cuda()
# net.eval()
# net = torchvision.models.resnet101(pretrained=True)
# state_dict =torch.load('densenet121-241335ed.pth')
# own_state = net.state_dict()
#
# for name, param in state_dict.items():
#     print name
#     if name not in own_state:
#         print "not"+name
#         pass
#     if isinstance(param, Parameter):
#         # backwards compatibility for serialized parameters
#         param = param.data
#     try:
#         own_state[name].copy_(param)
#         missing = set(own_state.keys()) - set(state_dict.keys())
#     except:
#         print 'unable to load'+name

# net = densenet121(False)
net = torch.nn.DataParallel(net,device_ids=[0,1])

net.cuda()
net.load_state_dict(torch.load('pretrain/epoch_02.31319937705'))

cudnn.benchmark = True

focalloss = FocalLoss(alpha=None,gamma=0.5,size_average=True)
#steps when lr decays
steps = [20,40,60]

optimizer = optim.SGD(net.parameters(),lr = lr,momentum=momentum,weight_decay=weight_decay )

for epoch in range(start_epoch,max_epoch):
    print 'epoch:',epoch
    epoch_loss = 0
    net.train()

    for batch_idx,(images,target) in enumerate(trainloader):

        images = Variable(images.cuda())
        target = Variable(target.cuda())
        out = net(images)
        # loss = F.cross_entropy(out,target)

        loss = focalloss(out,target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_loss = epoch_loss*(batch_idx/(batch_idx+1.0))+loss.data[0]*(1.0/(batch_idx+1.0))
        if batch_idx%50 == 0:
            print 'batch:',batch_idx,'batch_loss: %.4f'%(loss.data[0]),"epoch_loss: %.4f"%(epoch_loss)
        if batch_idx%1000 == 999:

            torch.save(net.state_dict(), 'pretrain/' + 'epoch_' + str(epoch) + str(epoch_loss))

    if epoch in steps:
        lr*=0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
