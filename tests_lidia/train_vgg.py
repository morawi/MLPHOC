from __future__ import print_function, division

import copy
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

from datasets.load_washington_dataset import WashingtonDataset
from scripts.data_transformations import PadImage
from utils import globals

warnings.filterwarnings("ignore")

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Washington Dataset Loading

image_transfrom = transforms.Compose([PadImage((globals.MAX_IMAGE_WIDTH, globals.MAX_IMAGE_HEIGHT)),
                                      transforms.ToPILImage(),
                                      transforms.Scale((globals.NEW_W, globals.NEW_H)),
                                      transforms.ToTensor()
                                      ])

wg_train_dataset = WashingtonDataset(txt_file='datasets/washingtondb-v1.0/ground_truth/word_labels.txt',
                                       root_dir='datasets/washingtondb-v1.0/data/word_images_normalized',
                                       train=True,
                                       transform=image_transfrom,
                                       non_alphabet=False)

wg_test_dataset = WashingtonDataset(txt_file='datasets/washingtondb-v1.0/ground_truth/word_labels.txt',
                                       root_dir='datasets/washingtondb-v1.0/data/word_images_normalized',
                                       train=False,
                                       transform=image_transfrom,
                                       non_alphabet=False)

train_dataloader = DataLoader(wg_train_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

test_dataloader = DataLoader(wg_test_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

dataloaders = {'train': train_dataloader, 'test': test_dataloader}

dataset_sizes = {len(wg_train_dataset), len(wg_test_dataset)}
class_names = {1,2,3,4,5,6,7,8,9,10}
epochs=2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2)