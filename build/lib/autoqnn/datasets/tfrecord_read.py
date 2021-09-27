import os
import tfrecord
def create_index(file_dir,index_dir):
    '''
    Examples:
    train_dir="/NVME1/imagenet/tf_records/train/"
    train_index_dir="/NVME1/imagenet/tf_records/train_index/"
    val_dir="/NVME1/imagenet/tf_records/val/"
    val_index_dir="/NVME1/imagenet/tf_records/val_index/"
    create_index(train_dir,train_index_dir)
    create_index(val_dir,val_index_dir)
    '''
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
