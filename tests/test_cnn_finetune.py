"""
Main @author: malrawi

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from cnn_finetune import make_model
from utils.some_functions import find_mAP_QbS, find_mAP_QbE #, add_weights_of_words
from utils.some_functions import word_str_moment, phoc_confusion_matrix, word_similarity_metric, count_model_parameters #test_varoius_dist, 
from datasets.get_datasets import get_datasets,  get_dataloaders, get_transforms
from sklearn.metrics import accuracy_score
from utils.pwdistance import accuracy_score as my_accuracy_score
from utils.prediction import predict_labels # can be used when targets no available


def test_cnn_finetune(cf):
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_transform = get_transforms(cf)
    train_set, test_set, test_per_data = get_datasets(cf, image_transform)
    train_loader, test_loader, per_data_loader = get_dataloaders(cf, train_set, test_set, test_per_data)
               
    
# build the model
    model = make_model(
        cf.model_name,
        pretrained = cf.pretrained,
        num_classes = len(test_set[1][1]), #train_set.num_classes(),
        input_size = cf.input_size, 
        dropout_p = cf.dropout_probability,
    )
    model = model.to(device)

# define the loss function
    if cf.loss == 'BCEWithLogitsLoss':
#        if cf.use_weight_to_balance_data:         
#            criterion = nn.BCEWithLogitsLoss(reduce = None) # size_average=True, reduction='sum')
#        else: 
        criterion = nn.BCEWithLogitsLoss()          
    elif cf.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else: 
        criterion = nn.MSELoss()
        
# the optimizer
    optimizer = optim.SGD(model.parameters(), 
                          lr = cf.learning_rate, 
                          momentum = cf.momentum,
                          nesterov = cf.use_nestrov_moment,
                          weight_decay = cf.weight_decay,
                          dampening = cf.damp_moment if not(cf.use_nestrov_moment) else 0
                          )    
    print('--- Total no. of params in model ', count_model_parameters(model), '-------')
    
    def train(epoch):
        total_loss = 0; total_size = 0
        model.train()
        for batch_idx, (data, target, word_str, weight) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) 
            optimizer.zero_grad()
            output = model(data)                         
            if cf.loss=='MSELoss':  output= torch.nn.Sigmoid(output)                                       
            loss = criterion(output.float(), target.float())
            total_loss += loss.item()
            total_size += data.size(0) # accumulating the size by accumulating batc sizes
            loss.backward()
            optimizer.step()
            if batch_idx % cf.batch_log == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Average loss: {:.7f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), total_loss / total_size))
        del data, target



    def test(test_loader):                       
        
        word_str_all = (); total_loss = 0; total_size=0;                   
        pred_all = torch.tensor([], dtype=torch.float32, device=device)
        target_all = torch.tensor([], dtype=torch.float32, device=device)
                
        model.eval()  # sets the Dropout (and Normalization layers, if any) in evaluation mode
        with torch.no_grad():
            for data, target, word_str, weight in test_loader: # weight is trivial here
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.float(), target.float())
                total_loss += loss.item()
                total_size += data.size(0)                
                output = F.sigmoid(output)
                pred = output.data  
                pred_all = torch.cat((pred_all, pred), 0) # Accumulate from batches to one variable (##_all)
                target_all = torch.cat( (target_all, target), 0)
                word_str_all = word_str_all + word_str     

        
        result = {'word_str':word_str_all, 'pred': pred_all, 'target': target_all}
        if cf.encoder=='phoc' or cf.encoder=='varphoc': 
            
            print('Sklearn accuracy: ' , accuracy_score(target_all.cpu().numpy(), pred_all.cpu().numpy().round(), normalize=True) ) # sklearn accu, round() is needed here to convert the pred to binary     
            result['correct'], _, _,_ = my_accuracy_score(target_all.cpu().numpy(), pred_all.cpu().numpy(), 
                  labels_true = word_str_all, normalize=False,  diagnostics=False)            
           # predicted_labels = predict_labels(target_all.cpu().numpy(), pred_all.cpu().numpy(), word_str_all)
            
        elif cf.encoder=='chars2vec' or cf.encoder=='phonetic_vec':             

            result['correct'], _, _,_ = my_accuracy_score(target_all.cpu().numpy(), pred_all.cpu().numpy(), 
                  labels_true = word_str_all, normalize=False,  diagnostics=True)
                  
        
        if cf.task_type == 'word_spotting':
            result['mAP_QbS']  = find_mAP_QbS(result, cf)
            result['mAP_QbE'] = find_mAP_QbE(result, cf)
        else: 
             result['conf_mat'] = phoc_confusion_matrix(result, cf)
            
        if cf.print_accuracy == True:
            print( 'QbS ',  result['mAP_QbS'], " QbE ",  result['mAP_QbE'], " ")    
            print('\n Test set:\t Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
            total_loss/total_size,  result['correct'], pred_all.size()[0], 100. * result['correct'] / pred_all.size()[0]))
        
        del pred_all, target_all, data
        result.clear() # there seems to be an issue with the memory, even if not receiving 'result'
        
        return result # to be used in case we want to try different distances later
    
    
    
    def test_moment_and_word_sim(result, test_set):
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
                print(word_str_mom, " ", end="")
                print(word_similarity)
                
    
    
    # perroms testing for multidata, bu teating each data separately
    def splitted_sets_testing():
        if len(test_per_data.items())==1 : return 
        for item in test_per_data.items():
            print('Split the merged data: Per-script/data testing  --------', item[0]); 
            result = test(per_data_loader[item[0]]) 
        return result

    
    print('Chance level performance \n');  
    test(test_loader)  
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cf.lr_milestones , gamma= cf.lr_gamma) 
    # Training
    for epoch in range(1, cf.epochs + 1):
        scheduler.step();  
        print("lr = ", scheduler.get_lr(), end="") # to be used with MultiStepLR
        train(epoch)           
        if not(epoch % cf.testing_print_frequency):
            print('&&&&&&-------- all data testing ------ &&&&&&')
            test(test_loader)
            splitted_sets_testing()            
                
    print('&&&&&&&&&&&&&&&&&&    Retrieval testing on all data  &&&&&&&&&&&&&&&')        
    test(test_loader) # performance after training is finished

    result_final = splitted_sets_testing()
    
   
    
    return result_final
    
     # test_moment_and_word_sim(result)
# '''   depreciated
# if cf.use_weight_to_balance_data:
#                weight = weight.to(device)
#                output = torch.mul(weight, torch.transpose(output.double(), 0, 1) )
#                output = torch.transpose(output, 0, 1)
#                target = torch.mul(weight, torch.transpose(target.double(), 0, 1) )
#                target = torch.transpose(target, 0, 1)
#'''            
## during training     
#if cf.use_weight_to_balance_data:
#                weight = weight.to(device)                
#                loss = criterion( output, target )
#                loss = torch.mul(weight, torch.transpose(loss, 0, 1) )
#                loss= torch.mean(loss)    

## during testing     
#if cf.use_weight_to_balance_data:
#                    total_loss += torch.mean(loss).item()