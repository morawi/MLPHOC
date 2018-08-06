"""
Structure Based on:
- https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
- https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from cnn_finetune import make_model

# from utils import globals
from utils.some_functions import find_mAP, binarize_the_output, find_mAP_QbS, find_mAP_QbE
from datasets.load_washington_dataset import WashingtonDataset
from datasets.load_ifnenit_dataset import IfnEnitDataset
from datasets.load_WG_IFN_dataset import WG_IFN_Dataset
from datasets.load_iam_dataset import IAM_words
from scripts.data_transformations import PadImage, ImageThinning
from utils.some_functions import word_str_moment, word_similarity_metric #test_varoius_dist, 



def test_cnn_finetune(cf):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
#    mean = (0.5, 0.5, 0.5) 
#    std = (0.25, 0.25 , 0.25)     

    # Image transformations
    if cf.pad_images:
        pad_image = PadImage((cf.MAX_IMAGE_WIDTH, cf.MAX_IMAGE_HEIGHT))
    
    '''  I am not sure if these transforms.ToPILImage() are needed?
        we need to inspect everyting, and I would really like to 
        re-wriet these if-stmnt in a btter, and clearer form
     '''

    thin_image = ImageThinning(p = cf.thinning_threshold)


    if cf.resize_images:
        if cf.pad_images:
            image_transfrom = transforms.Compose([
                  #  thin_image,
                    pad_image,
                    transforms.ToPILImage(),
                    transforms.Scale((cf.input_size[0], cf.input_size[1])),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
  #transforms.Normalize(mean, std),
                      ])
        else:
            image_transfrom = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Scale((cf.input_size[0], cf.input_size[1])),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    # transforms.Normalize(mean, std),
                    ])
    else:
        if cf.pad_images:
            image_transfrom = transforms.Compose([
                    pad_image,
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    # transforms.Normalize(mean, std),
                    ])
        else:
            image_transfrom = transforms.Compose([transforms.ToTensor(),
                                                  # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                  # transforms.Normalize(mean, std),
                                                  ])

    if cf.dataset_name == 'WG':
        print('...................Loading WG dataset...................')
        train_set = WashingtonDataset(cf, train=True, transform=image_transfrom)
        test_set = WashingtonDataset(cf, train=False, transform=image_transfrom, 
                            data_idx =train_set.data_idx, complement_idx = True)
    
    elif cf.dataset_name == 'IFN':
        # TODO
        print('...................Loading IFN dataset...................')        
        train_set = IfnEnitDataset(cf, train=True, transform=image_transfrom)
        test_set = IfnEnitDataset(cf, train=False, transform=image_transfrom, 
                            data_idx =train_set.data_idx, complement_idx = True)
        
    elif cf.dataset_name =='WG+IFN': 
        print('...................IFN & WG datasets ---- The multi-lingual PHOCNET')        
        ## TODO
        train_set = WG_IFN_Dataset(cf, train=True, transform=image_transfrom)
        test_set = WG_IFN_Dataset(cf, train=False, transform=image_transfrom, 
                                  data_idx_WG = train_set.data_idx_WG, 
                                  data_idx_IFN = train_set.data_idx_IFN, 
                                        complement_idx = True)
    elif cf.dataset_name =='IAM':
        print('...................Loading IAM dataset...................') 
        train_set = IAM_words(cf, mode='train', transform = image_transfrom, augmentation=True )
        test_set = IAM_words(cf, mode='test', transform = image_transfrom, augmentation=True )
        
        print('IAM IAM')
        # plt.imshow(train_set[29][0], cmap='gray'); plt.show()
               
     
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cf.batch_size_train,
                                  shuffle=cf.shuffle, num_workers=cf.num_workers)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cf.batch_size_test,
                                  shuffle= False, num_workers=cf.num_workers)
   

    model = make_model(
        cf.model_name,
        pretrained=cf.pretrained,
        num_classes= train_set.num_classes(),
        input_size=(cf.input_size[0], cf.input_size[1]),
        dropout_p=cf.dropout_probability,
    )
    model = model.to(device)

   
    if cf.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss() # size_average=True, reduction='sum')
    elif cf.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else: 
        criterion = nn.MSELoss()
        

    optimizer = optim.SGD(model.parameters(), lr=cf.learning_rate, momentum=cf.momentum)    

    def train(epoch):
        total_loss = 0
        total_size = 0
        model.train()
        for batch_idx, (data, target, word_str) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if cf.loss=='MSELoss': 
                output= F.sigmoid(output)
                loss = criterion(output.float(), target.float())
            else: 
                loss = criterion(output.float(), target.float())
            total_loss += loss.item()
            total_size += data.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % cf.batch_log == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.7f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), total_loss / total_size))


    def test(test_loader):               
        
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
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()                
                # Accumulate from batches to one variable (##_all)
                pred_all = torch.cat((pred_all, pred), 0)
                target_all = torch.cat((target_all, target), 0)
                word_str_all = word_str_all + word_str        
        
        test_loss /= len(test_loader.dataset)             
        result = {'word_str':word_str_all,'pred': pred_all,
                  'target': target_all}
        
        '''
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset)*pred.size()[1],
            100. * correct / (len(test_loader.dataset)*pred.size()[1] )))   
        '''
   
        # mAP_QbE, mAP_QbS = find_mAP(result, cf)               
       
        mAP_QbS = find_mAP_QbS(result, cf)
        mAP_QbE = find_mAP_QbE(result, cf)
        print(  mAP_QbS, "  ",  mAP_QbE, " ", end="")        
        result['mAP_QbE'] = mAP_QbE
        result['mAP_QbS'] = mAP_QbS
        
        
        return result # to be used in case we want to try different distances later
    
    def test_moment_and_word_sim(result):
        print('--------------Moment and similarity statistics ----------------- ')
        no_of_samples  = len(result['word_str'])
        step = 200
        # sampler_selector_sizes = [i for i in range(no_of_samples-100, 200, -step )]            
        sampler_selector_sizes = [no_of_samples - 100]
        no_moment_iterations = 10
        for sample_size in sampler_selector_sizes:
            word_str_mom = []; word_similarity=[]
            for  xxnn in range(0, no_moment_iterations):            
                sample_idx = np.random.permutation(np.arange(1, no_of_samples))[:sample_size]                     
                if len(sample_idx) ==0:  
                    exit('exiting function get_the_sampler(), sample_idx size is 0')    
                my_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)  
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=cf.batch_size_test,
                                  shuffle= False, num_workers=cf.num_workers, sampler=my_sampler)
                res = test(test_loader)
                word_str_mom = word_str_moment(res['word_str'])
                word_similarity = word_similarity_metric(res['word_str'])
                # print('word Moment =--: ', word_str_mom, end="")
                # print('word Similarity =--: ', word_similarity, '\n')
                print(  word_str_mom, " ", end="")
                print( word_similarity)


                
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cf.lr_milestones , gamma= cf.lr_gamma) 
    # print('PHOC length', train_set.)
    print('Chance level performance \n');  
    test(test_loader) # nice to know the performance prior to training
    for epoch in range(1, cf.epochs + 1):
        scheduler.step();  print("lr = ", scheduler.get_lr(), " ", end ="") # to be used with MultiStepLR
        train(epoch)           
        if not(epoch % cf.testing_print_frequency):
            test(test_loader)
            
    result = test(test_loader)
    test_moment_and_word_sim(result)
    
    return result, train_set, test_set, train_loader, test_loader # returned to be checked from command console, this is provisary
    
    
    
