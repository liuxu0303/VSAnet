import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
from PIL import Image
import random

def load_as_float(path):
    img = imread(path).astype(np.float32)
    return img

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/train/0000000.png
        root/train/0000000.txt
        ...
        root/test/0000000.png
        root/test/0000000.txt
        ...
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, transform=None, dataset='VSA300'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.sample_path = self.root + '/train' if train else self.root + '/test'
        self.transform = transform
        self.dataset = dataset
        self.flag = train
        self.crawl_folders() 
        

    def crawl_folders(self):
        
        sample_set = []
        imgs = sorted(self.sample_path.files('*.png'))
        for i in range(len(imgs)):
            img = imgs[i]
            info = img[:-4]+'.txt'
            info = np.genfromtxt(info, delimiter=' ', dtype=float).astype(np.float32)
            label = info[0]  # the label of VSA
            classid = info[1]
                
            sample = {'img': img, 'label': label, 'classid': classid}
            sample_set.append(sample)
                
        if self.flag:            
            random.shuffle(sample_set)
            
        self.samples = sample_set

    def __getitem__(self, index):
        sample = self.samples[index]
        img = Image.open(sample['img'])
        
        arealabel = sample['label']    # the ground truth VSA label
        classid = sample['classid']
        classification = 0  # the ground truth category label
        Vlength = 1908  # the length of set_V
        
        areamax = 0.      # preset the reference maximum boundary
#        areamin = 0.
        set_V = np.zeros(int(Vlength))  # the set V_i of all ground truth VSA of the i-th category
        
        if classid == 56:
            classification = 0  
            set_V = np.load('./path to V_chair.npy')
            areamax = 0.9
#            areamin = 0.1
        if classid == 57:
            classification = 1
            set_V = np.load('./path to V_sofa.npy')
            areamax = 3.3
#            areamin = 0.4
        if classid == 59:
            classification = 2
            set_V = np.load('./path to V_bed.npy')
            areamax = 3.7
#            areamin = 0.4
        if classid == 62:
            classification = 3
            set_V = np.load('./path to V_tvmonitor.npy')
            areamax = 0.26
#            areamin = 0.04
        if classid == 41:
            classification = 4
            set_V = np.load('./path to V_cup.npy')
            areamax = 0.05
#            areamin = 0.005
        if classid == 39:
            classification = 5
            set_V = np.load('./path to V_bottle.npy')
            areamax = 0.05
#            areamin = 0.003
        if classid == 66:
            classification = 6
            set_V = np.load('./path to V_keyboard.npy')
            areamax = 0.07
#            areamin = 0.007            
        if classid == 64:
            classification = 7
            set_V = np.load('./path to V_mouse.npy')
            areamax = 0.0071
#            areamin = 0.003        
        if classid == 63:
            classification = 8
            set_V = np.load('./path to V_laptop.npy')
            areamax = 0.173
#            areamin = 0.034
        
        classification = np.array(classification)
        classlabel = classification.astype(np.long)
        arealabel = arealabel.astype(np.float32)
        set_V = np.pad(set_V,(0,Vlength-len(set_V)),'constant', constant_values=(0,0))
        set_V = set_V.astype(np.float32)
        areamax = np.array(areamax)
        areamax = areamax.astype(np.float32)
        
        if self.transform is not None:
            img = self.transform(img)
            
        imgids = sample['img']

        return img, arealabel, classlabel, set_V, areamax, imgids

     
    def __len__(self):
        return len(self.samples)

