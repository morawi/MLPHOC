from __future__ import print_function, division

import warnings

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from scripts.data_transformations import PadImage
from utils import globals
from datasets.load_washington_dataset import WashingtonDataset
from datasets.load_ifnenit_dataset import IfnEnitDataset
import logging, sys

warnings.filterwarnings("ignore")

# Test the Washington Dataset Loading

def test_dataload(cf):
    logger = logging.getLogger('test_dataloader_wg')

    # Image transformations
    if cf.pad_images:
        pad_image = PadImage((globals.MAX_IMAGE_WIDTH, globals.MAX_IMAGE_HEIGHT))

    if cf.resize_images:
        if cf.pad_images:
            image_transfrom = transforms.Compose([pad_image,
                                           transforms.ToPILImage(),
                                           transforms.Scale((cf.input_size[0], cf.input_size[1])),
                                           transforms.ToTensor()])
        else:
            image_transfrom = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Scale((cf.input_size[0], cf.input_size[1])),
                                           transforms.ToTensor()])
    else:
        if cf.pad_images:
            image_transfrom = transforms.Compose([pad_image,
                                       transforms.ToTensor()])
        else:
            image_transfrom = transforms.ToTensor()

    if cf.dataset_name == 'WG':
        input_dataset = WashingtonDataset(cf, transform=image_transfrom)

    elif cf.dataset_name == 'IFN':
        input_dataset = IfnEnitDataset(cf, transform=image_transfrom)
    else:
        logger.fatal('The dataset \'%s\' is unknown. Use: [WG, IFN]', cf.dataset_name)
        sys.exit(0)

    dataloader = DataLoader(input_dataset, batch_size=cf.batch_size,
                            shuffle=cf.shuffle, num_workers=cf.num_workers)

    for i in range(len(input_dataset)):
        plt.figure(i);
        plt.xticks([]);
        plt.yticks([])
        data, target = input_dataset[i]
        plt.imshow(data.numpy()[0, :, :], 'gray')
        plt.show();

        if i == 2: break

