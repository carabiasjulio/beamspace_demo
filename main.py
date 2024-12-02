import torch
import yaml
import pathlib
from dataset import MyCustomDataset


def main():
    # read the configuration file
    with open('./conf.yaml','r') as f:
        conf = yaml.safe_load(f)
    conf['dataset']['root_dir'] = pathlib.Path(conf['dataset']['root_dir'])

    # create the dataloader of the train and test splits
    dataset = MyCustomDataset(conf['dataset']['root_dir'])
    # obtain a train and valid split for the dataset and define dataloaders
    torch.manual_seed(conf['dataset']['seed'])
    train_set, val_set = torch.utils.data.random_split(dataset,conf['dataset']['train_valid_split_sizes'])
    print(f'\nTrain and valid set lengths: {len(train_set),len(val_set)}\n')

    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=conf['dataset']['batch_size'],
                                            shuffle=True,
                                            num_workers=conf['dataset']['num_workers'],
                                            pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=conf['dataset']['batch_size'],
                                            shuffle=False,
                                            num_workers=conf['dataset']['num_workers'],
                                            pin_memory=True)
    
    # Create an iterator from the train_loader
    train_loader_iter = iter(train_loader)

    # Get one example instance (first batch)
    example_instance = next(train_loader_iter)


if __name__ == '__main__':
    main()