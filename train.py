import cv2
import torch
from utils import Loader,Indoor
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from Network import densenet121,focal_loss
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter


transform1 = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(),transforms.RandomCrop((448,448)),transforms.Normalize(mean= ( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
transform2 = transforms.Compose([transforms.ToTensor,transforms.Normalize(mean=( 0.44731586 ,0.47744268,  0.49484214),std = ( 0.225 , 0.225 , 0.225))])
trainset = Loader(root = '/home/zhou/ai_challenger_scene_train_20170904/scene_train_images_20170904',list_file='/home/zhou/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json',train = True,transform = transform1)
testset = Loader(root = '/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',list_file='/home/zhou/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json',train=False,transform=transform2)
# trainset = Indoor(root = '/home/zhou/Images',list_file = 'TrainImages.txt',train= True,transform=transform)
# testset = Indoor(root = '/home/zhou/Images',list_file = 'TestImages.txt',train= False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size = 16,shuffle = True,num_workers = 4)
testloader = torch.utils.data.DataLoader(testset,batch_size = 8,shuffle = False,num_workers = 4)


#hyper parameters

lr = 0.01
weight_decay = 1e-4
start_epoch = 0
max_epoch = 50
resume = False



net = densenet121(False)
net = torch.nn.DataParallel(net,device_ids=[0,1])
net.cuda()
if resume:
    net.load_state_dict(torch.load('dilation/epoch_33_top1_0.566_top3_0.759.pkl'))
    # state_dict =torch.load('densenet121-241335ed.pth')
    # own_state = net.state_dict()
    # try:
    #     print own_state.keys()
    #     for name, param in state_dict.items()[:-1]:
    #         name = name
    #         if 'module.'+name not in own_state:
    #             raise KeyError('unexpected key "{}" in state_dict'.format(name))
    #         if isinstance(param, Parameter):
    #             # backwards compatibility for serialized parameters
    #             param = param.data
    #         own_state['module.'+name].copy_(param)
    #     missing = set(own_state.keys()) - set(state_dict.keys())
    #     if len(missing) > 0:
    #         raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    #     print 'load model successfully'
    # except:
    #     pass



cudnn.benchmark = True
steps = [5,10,20]

optimizer = optim.SGD(net.parameters(),lr = lr,momentum=0.9,weight_decay=weight_decay )

for epoch in range(start_epoch,max_epoch):
    print 'epoch:',epoch
    epoch_loss = 0
    net.train()
    for batch_idx,(images,target) in enumerate(trainloader):
        images = Variable(images.cuda())
        target = Variable(target.cuda())
        out = net(images)
        loss = F.cross_entropy(out,target)
        # loss = focal_loss(out,target,2)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_loss = epoch_loss*(batch_idx/(batch_idx+1.0))+loss.data[0]*(1.0/(batch_idx+1.0))
        if batch_idx%50 == 0:
            print 'batch:',batch_idx,'batch_loss: %.4f'%(loss.data[0]),"epoch_loss: %.4f"%(epoch_loss)
    if epoch in steps:
        lr*=0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #test
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
        counter += 8

    print batch_id
    torch.save(net.state_dict(),'Indoor/'+'epoch_'+repr(epoch)+'_top1_'+str(float(top1)/counter)[:5]+'_top3_'+str(float(top3)/counter)[:5]+'.pkl')
    print 'Top3 accuracy:' ,float(top3)/counter,' Top1 accuracy:' ,float(top1)/counter


