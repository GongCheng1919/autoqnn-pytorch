import torch
import torchvision
import torchvision.transforms as transforms
from .autoaugment import CIFAR10Policy
from .cutout import Cutout

cifar100_means = [129.3, 124.1, 112.4]
cifar100_stds = [68.2, 65.4, 70.4]
cifar10_means = [125.30691805, 122.95039414, 113.86538318]
cifar10_stds = [62.99321928, 62.08870764, 66.70489964]

def get_cifar_dataset(root='/workspace/datasets/cifar10',dataset = 'cifar10',autoaugment=True):

    __dataset_obj__={'cifar10':torchvision.datasets.CIFAR10,'cifar100':torchvision.datasets.CIFAR100}
    dataset_obj = __dataset_obj__[dataset]
    # Dataset
    if autoaugment:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                             transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                             Cutout(n_holes=1, length=16),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        # transform_train = transforms.Compose([
        #         transforms.Pad(4, padding_mode='reflect'),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(32),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[x / 255.0 for x in cifar100_means],
        #                                      std=[x / 255.0 for x in cifar100_stds])
        #     ])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])

    # transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[x / 255.0 for x in cifar100_means],
    #                                      std=[x / 255.0 for x in cifar100_stds])])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = dataset_obj(root=root, 
                            train=True, 
                            download=True, 
                            transform=transform_train)
    testset = dataset_obj(root=root, 
                           train=False, 
                           download=True, 
                           transform=transform_test)
    return trainset, testset

# example:
#     trainset, testset = get_cifar10_dataset()

def get_cifar_dataloader(train_batch_size=500,val_batch_size=100, num_workers=4, autoaugment=True,
                         root='/workspace/datasets/cifar10',
                         dataset = 'cifar10'):

    
    trainset, testset = get_cifar_dataset(root=root,dataset=dataset,autoaugment=autoaugment)
    
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=train_batch_size, 
                                              shuffle=True, 
                                              num_workers=num_workers) 
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=val_batch_size, 
                                             shuffle=False, 
                                             num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader,testloader,classes

# example:
#     trainloader,testloader,classes = get_cifar10_dataloader()