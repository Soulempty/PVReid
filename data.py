from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing,ToGraying
from utils.RandomSampler import RandomSampler
from opt import opt
import os.path as osp
import glob
import copy
import random
import os

import re
    
class Data():
    def __init__(self,data="veri",size=(288, 288),sampler="triple",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        train_transform = transforms.Compose([
            transforms.Resize(size, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            RandomErasing(probability=0.5, mean=mean)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        Dataset = {'veri':VeRi,"vehicleid":VehicleID,'market':Market1501,'msmt':MSMT17}
        
        self.trainset = Dataset[data](train_transform, 'train', opt.data_path)
        self.testset = Dataset[data](test_transform, 'test', opt.data_path)
        self.queryset = Dataset[data](test_transform, 'query', opt.data_path)
        self.nums = self.trainset.len
        
        if sampler==None:
            self.train_loader = dataloader.DataLoader(self.trainset,batch_size=opt.batchsize, shuffle = True,num_workers=4,pin_memory=True)
        elif sampler=="triple":
            self.train_loader = dataloader.DataLoader(self.trainset,sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                     batch_image=opt.batchimage),
                                     batch_size=opt.batchid * opt.batchimage, num_workers=4,
                                     pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, shuffle = False, num_workers=4, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, shuffle = False, num_workers=4,
                                    pin_memory=True)
class VeRi(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):
        self.transform = transform
        self.loader = default_loader
        self.len = 0
        self.data = {'train':'image_train','test':'image_test','query':'image_query'}

        files = glob.glob(osp.join(data_path,self.data[dtype],'*.jpg'))
        id2label = {}
        pattern = re.compile(r'([-\d]+)_c([\d]+)')
        self.imgs = []
        self.pids = []
        self.camids = []
        for f in files:
            img_name=osp.basename(f)
            pid, camid = map(int, pattern.search(img_name).groups())
            if pid == -1: 
                continue
            camid -= 1
            if dtype=='train':
                if pid not in id2label:
                    id2label[pid]=len(id2label)
            else:
                if pid not in id2label:
                    id2label[pid]=pid                    
            label_id = id2label[pid]
            self.imgs.append(f)
            self.pids.append(label_id)
            self.camids.append(camid)
        self.len = len(id2label)
    def __getitem__(self, index):
        path = self.imgs[index]
        pid = self.pids[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid

    def __len__(self):
        return len(self.imgs)

class VehicleID(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):
   
        self.transform = transform
        self.loader = default_loader
        self.len = 0
        self.img_path = data_path+'/image'
        self.data = {'train':'/train_test_split/train_list.txt','test':'/train_test_split/gallery1600.txt','query':'/train_test_split/query1600.txt'}
        lines = open(data_path+self.data[dtype]).readlines()
        
        id2label = {}
        self.imgs = []
        self.pids = []
        self.camids = []
        for line in lines:
            img_name = line.strip().split()[0]
            f = self.img_path+"/"+img_name+".jpg"
            pid = int(line.strip().split()[1])
            if pid == -1: 
                continue
            camid = 0
            if dtype=='train':
                if pid not in id2label:
                    id2label[pid]=len(id2label)
            else:
                if pid not in id2label:
                    id2label[pid]=pid                    
            label_id = id2label[pid]
            self.imgs.append(f)
            self.pids.append(label_id)
            self.camids.append(camid)    
        self.len = len(id2label)
    def __getitem__(self, index):
        path = self.imgs[index]
        pid = self.pids[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid
    def __len__(self):
        return len(self.imgs)
    
class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.len = 0
        self.data = {'train':'bounding_box_train','test':'bounding_box_test','query':'query'}
        files = glob.glob(osp.join(data_path,self.data[dtype],'*.jpg'))

        id2label = {}
        self.imgs = []
        self.pids = []
        self.camids = []
        for f in files:
            img_name=osp.basename(f)
            pid, camid = int(img_name.split('_')[0]),int(img_name.split('_')[1][1])
            if pid == -1: 
                continue
            camid -= 1
            if dtype=='train':
                if pid not in id2label:
                    id2label[pid]=len(id2label)
            else:
                if pid not in id2label:
                    id2label[pid]=pid                    
            label_id = id2label[pid]
            self.imgs.append(f)
            self.pids.append(label_id)
            self.camids.append(camid)
        self.len = len(id2label)
    def __getitem__(self, index):
        path = self.imgs[index]
        pid = self.pids[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid

    def __len__(self):
        return len(self.imgs)

    
class MSMT17(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.len = 0
        self.data = {'train':'train','test':'gallery','query':'query'}
        files = glob.glob(osp.join(data_path,self.data[dtype],'*.jpg'))

        id2label = {}
        self.imgs = []
        self.pids = []
        self.camids = []
        for f in files:
            img_name=osp.basename(f)
            pid, camid = int(img_name.split('_')[0]),int(img_name.split('_')[2])
            if pid == -1: 
                continue
            camid -= 1
            if dtype=='train':
                if pid not in id2label:
                    id2label[pid]=len(id2label)
            else:
                if pid not in id2label:
                    id2label[pid]=pid                    
            label_id = id2label[pid]
            self.imgs.append(f)
            self.pids.append(label_id)
            self.camids.append(camid)
        self.len = len(id2label)
    def __getitem__(self, index):
        path = self.imgs[index]
        pid = self.pids[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid

    def __len__(self):
        return len(self.imgs)