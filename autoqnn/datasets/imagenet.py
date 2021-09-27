import shutil,os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import tfrecord
from tfrecord.torch.dataset import MultiTFRecordDataset
MultiTFRecordDataset.__len__=lambda self:self.length
import cv2
from PIL import Image
from .base import DataLoaderX, data_prefetcher

def mvVal2ValFolders(data_path):
    val_txt="/data/gongcheng/imagenet/ilsvrc12/val.txt"
    val_df=pd.read_csv(val_txt,sep='\s+',header=None,names=["images","labels"])
    filename2id={name:ids for name,ids in zip(val_df["images"],val_df["labels"])}
    synsets="/data/gongcheng/imagenet/ilsvrc12/synsets.txt"
    synset_words="/data/gongcheng/imagenet/ilsvrc12/synset_words.txt"
    synsets_df=pd.read_csv(synsets,sep='\s+',header=None,names=["folder_names"])
    id2foldername=np.array(synsets_df["folder_names"])
    synset_words_df=pd.read_csv(synset_words,sep='\s+',header=None,names=["folder_names","class_name"])
    val_path = os.path.join(data_path, 'val')
    n=0
    for img,ids in filename2id.items():
        folder_name=id2foldername[ids]
        folder_path=os.path.join(val_path,folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
    #         print("\n\rmkdir %s"%folder_path)
        # move
        src=os.path.join(val_path, img)
        dst=os.path.join(folder_path, img)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.move(src,dst)
            print("\t\r(%.2f%%) %s ====> %s"%((n+1)/len(filename2id)*100,src, dst),end="")
        elif not os.path.isfile(src) and os.path.isfile(dst):
            print("\t\r(%.2f%%) %s exists"%((n+1)/len(filename2id)*100, dst),end="")
        else:
            raise ValueError("Can not find file %s or %s"%(src, dst))
        n+=1
        
def check_idx(train_dataset,val_dataset):
    for key,val in train_dataset.class_to_idx.items(): 
        assert(val_dataset.class_to_idx[key]==val)
        if val_dataset.class_to_idx[key]!=val:
            print((key,val),(key,val_dataset.class_to_idx[key]))
def to_one_hot(x,num_classes):
    return torch.nn.functional.one_hot(x,num_classes=num_classes)
        
def get_dataset(data_path,batch_size=256, workers=8, parse_type="torch",mean_std=([0,0,0],[1,1,1])):
    '''
    default mean=[0.485, 0.456, 0.406]
    default std=[0.229, 0.224, 0.225]
    '''
    __parse_type__={"caffe":([0.485, 0.456, 0.406],[1/255.0, 1/255.0, 1/255.0]),
                    "tf":([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    "torch":([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])}
    if parse_type not in __parse_type__ and len(mean_std)!=2:
        raise ValueError("Wrong parse_type (%s) and mean_std (%s) setting!"%(parse_type,"-".join(mean_std)))
    mean,std=__parse_type__[parse_type] if parse_type in __parse_type__ else mean_std 
    # data_path="/data/gongcheng/imagenet"
    batch_size=batch_size
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    # 数据预处理：normalize: - mean / std
    normalize = transforms.Normalize(mean=mean, std=std)
    
    # ImageFolder 一个通用的数据加载器
    train_dataset = datasets.ImageFolder(
        traindir,
        # 对数据进行预处理
        transforms.Compose([                      # 将几个transforms 组合在一起
            transforms.RandomResizedCrop(224),      # 随机切再resize成给定的size大小
            transforms.RandomHorizontalFlip(),    # 概率为0.5，随机水平翻转。
            transforms.ToTensor(),                # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray，
                                                  # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([ 
            # 重新改变大小为`size`，若：height>width`,则：(size*height/width, size)
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    check_idx(train_dataset,val_dataset)
    train_sampler = None
    # train 数据下载及预处理
    train_loader = DataLoaderX(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoaderX(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def create_index(file_dir,index_dir):
    if file_dir == index_dir:
        raise TypeError("index_dir should be different from file_dir")
    record_files=os.listdir(file_dir)
    for i,file in enumerate(record_files):
        file_path=os.path.join(file_dir,file)
        if os.path.isfile(file_path):
            index_path=os.path.join(index_dir,file+".index")
            print("\r[{0}/{1}] create {file_path} to {index_path}".format(i+1,len(record_files),file_path=file_path,index_path=index_path),end="")
            tfrecord.tools.tfrecord2idx.create_index(file_path,index_path)
    print("\n\r * Create index success!")

def check_index(tfrecord_dir,index_dir):
    assert(os.path.isdir(tfrecord_dir))
    if not os.path.isdir(index_dir):
        print("Create train index from {0} to {1}.".format(tfrecord_dir,index_dir))
        os.mkdir(index_dir)
        create_index(tfrecord_dir,index_dir)
# create_index(val_dir,val_index_dir)
def create_splits(file_dir):
    record_files=os.listdir(file_dir)
    splits={key:1./len(record_files) for key in record_files}
    return splits

def get_dat_from_tfrecord(tf_records_dir,batch_size=256,parse_type="torch",mean_std=([0,0,0],[1,1,1])):
    train_num=1281167
    val_num=50000
    train_dir = os.path.join(tf_records_dir,"train")
    train_index_dir = os.path.join(tf_records_dir,"train_index")
    val_dir = os.path.join(tf_records_dir,"val")
    val_index_dir = os.path.join(tf_records_dir,"val_index")
    check_index(train_dir,train_index_dir)
    check_index(val_dir,val_index_dir)
    train_config=dict(data_pattern=os.path.join(tf_records_dir,"train/{}"),
                      index_pattern=os.path.join(tf_records_dir,"train_index/{}.index"),
                      description={"image/encoded": "byte", "image/class/label": "int"},
                      splits=create_splits(train_dir))
    val_config=dict(data_pattern=os.path.join(tf_records_dir,"val/{}"),
                      index_pattern=os.path.join(tf_records_dir,"val_index/{}.index"),
                      description={"image/encoded": "byte", "image/class/label": "int"},
                      splits=create_splits(val_dir))
    __parse_type__={"caffe":([0.485, 0.456, 0.406],[1/255.0, 1/255.0, 1/255.0]),
                    "tf":([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    "torch":([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])}
    if parse_type not in __parse_type__ and len(mean_std)!=2:
        raise ValueError("Wrong parse_type (%s) and mean_std (%s) setting!"%(parse_type,"-".join(mean_std)))
    mean,std=__parse_type__[parse_type] if parse_type in __parse_type__ else mean_std 
    NUM_CLASS=1000
    train_transform = transforms.Compose([  
                transforms.RandomResizedCrop(224),  
                transforms.RandomHorizontalFlip(),    
                transforms.ToTensor(),              
                transforms.Normalize(mean=mean, std=std),
            ])
    val_transform = transforms.Compose([ 
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    # mean,std = ([0,0,0],[1,1,1])
    def decode_image(features,transform):
        # get BGR image from bytes
        img = cv2.imdecode(features["image/encoded"], -1)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform(img)
#         label = torch.nn.functional.one_hot(torch.tensor(features["image/class/label"]-1).squeeze(),
#                                                                     num_classes=NUM_CLASS)
        label = features["image/class/label"][0]-1
        return img,label
    train_dataset = MultiTFRecordDataset(**train_config, transform=lambda f: decode_image(f,transform=train_transform))
    train_dataset.length = 1281167
    train_loader = DataLoaderX(train_dataset, batch_size=batch_size)
    val_dataset = MultiTFRecordDataset(**val_config, transform=lambda f: decode_image(f,transform=val_transform))
    val_dataset.length = 50000
    val_loader = DataLoaderX(val_dataset, batch_size=batch_size)
    return train_loader, val_loader