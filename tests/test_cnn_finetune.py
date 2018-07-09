"""
Structure Based on:
- https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
- https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from cnn_finetune import make_model

from utils import globals
from datasets.load_washington_dataset import WashingtonDataset
from scripts.data_transformations import PadImage
from utils.some_functions import find_mAP


def test_cnn_finetune(cf):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Image transformations
    if cf.pad_images:
        pad_image = PadImage((globals.MAX_IMAGE_WIDTH, globals.MAX_IMAGE_HEIGHT))

    if cf.resize_images:
        if cf.pad_images:
            image_transfrom = transforms.Compose([pad_image,
                                                  transforms.ToPILImage(),
                                                  transforms.Scale((cf.input_size[0], cf.input_size[1])),
                                                  transforms.ToTensor(),
                                                  transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                                  ])
        else:
            image_transfrom = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Scale((cf.input_size[0], cf.input_size[1])),
                                                  transforms.ToTensor(),
                                                  transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
    else:
        if cf.pad_images:
            image_transfrom = transforms.Compose([pad_image,
                                                  transforms.ToTensor(),
                                                  transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
        else:
            image_transfrom = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    if cf.dataset_name == 'WG':

        train_set = WashingtonDataset(cf, train=True, transform=image_transfrom)
        test_set = WashingtonDataset(cf, train=False, transform=image_transfrom)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cf.batch_size_train,
                                  shuffle=cf.shuffle, num_workers=cf.num_workers)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cf.batch_size_test,
                                  shuffle=cf.shuffle, num_workers=cf.num_workers)

    num_classes = train_set.num_classes()

    model = make_model(
        cf.model_name,
        pretrained=cf.pretrained,
        num_classes=num_classes,
        input_size=(cf.input_size[0], cf.input_size[1]),
        dropout_p=cf.dropout_probability,
    )
    model = model.to(device)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=model.original_model_info.mean,
    #         std=model.original_model_info.std),
    # ])
    #
    # train_set = torchvision.datasets.CIFAR10(
    #     root='./data', train=True, download=True, transform=transform
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    # )
    # test_set = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_set, args.test_batch_size, shuffle=False, num_workers=2
    # )

    if cf.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(size_average=True)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=cf.learning_rate, momentum=cf.momentum)    

    def train(epoch):
        total_loss = 0
        total_size = 0
        model.train()
        for batch_idx, (data, target, word_str) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.float(), target.float())
            total_loss += loss.item()
            total_size += data.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.7f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), total_loss / total_size))


    def test():
        
        mAP_dist_metric = 'cosine'
        
        model.eval()
        
        test_loss = 0
        correct = 0
        pred_all = torch.tensor([], dtype=torch.float64, device=device)
        target_all = torch.tensor([], dtype=torch.float64, device=device)
        word_str_all = ()
        with torch.no_grad():
            for data, target, word_str in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output.float(), target.float()).item()
                output = F.sigmoid(output)
                pred = output.data
                pred = pred.type(torch.cuda.DoubleTensor)
                pred = pred.round()
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()                
               
                pred_all = torch.cat((pred_all, pred), 0)
                target_all = torch.cat((target_all, target), 0)
                word_str_all = word_str_all + word_str
               
        mAP_QbE,mAP_QbS = find_mAP(word_str_all, pred_all, target_all, mAP_dist_metric)       
        
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset)*pred.size()[1],
            100. * correct / (len(test_loader.dataset)*pred.size()[1] )))
        print('---- mAP(QbS)=', mAP_QbS, "---", 'mAP(QbE) = ', mAP_QbE, '----\n')
    
    lr_milestones = [100, 200, 500, 1000 ]  # it is better to move this in the config
    testing_frequency = 11  # it is better to move this in the config 
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones, gamma= .1) 
    print('Chance level performance \n');  test() #nice to know the performance prior to training
    for epoch in range(1, cf.epochs + 1):
        scheduler.step();  print("lr = ", scheduler.get_lr(), " ", end ="") # to be used with MultiStepLR
        train(epoch)           
        if not(epoch % testing_frequency):
            test()
            
    test()
    return train_set, test_set, train_loader, test_loader # returned to be checked from command console, this is provisary
    
    
    
