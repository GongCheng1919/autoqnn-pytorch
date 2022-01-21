import os
import tfrecord
from tfrecord.torch.dataset import MultiTFRecordDataset
MultiTFRecordDataset.__len__=lambda self:self.length
import cv2

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
