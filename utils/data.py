import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader

class Data:
  def __init__(self, root = '.data', batch_size = 128, valid_ratio = 0.9):
    self.root = root
    train_data = datasets.CIFAR10(root = root, train = True, download = True)
  
    # Compute means and standard deviations along the R,G,B channel
    self.means = train_data.data.mean(axis = (0,1,2)) / 255
    self.stds = train_data.data.std(axis = (0,1,2)) / 255
    self.batch_size = batch_size
    self.valid_ratio = valid_ratio


  def getTrainData(self, train_transforms = None):
    if train_transforms == None:
      train_transforms = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean = self.means, std = self.stds)
                ])

    train_data = datasets.CIFAR10(self.root, 
                      train = True, 
                      download = True, 
                      transform = train_transforms)

    train_iterator = DataLoader(train_data, batch_size = self.batch_size, shuffle = True) 

    return train_iterator

  def getTrainValData(self, train_transforms = None):
    if train_transforms == None:
      train_transforms = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean = self.means, std = self.stds)
                ])
                
    train_data = datasets.CIFAR10(self.root, 
                      train = True, 
                      download = True, 
                      transform = train_transforms)

    n_train_examples = int(len(train_data) * self.valid_ratio)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
    
    train_iterator = DataLoader(train_data, batch_size = self.batch_size, shuffle = True) 
    valid_iterator = DataLoader(valid_data, batch_size = self.batch_size, shuffle = False)   

    return train_iterator, valid_iterator


  def getTestData(self):
    test_transforms = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean = self.means, std = self.stds)
                      ])
    test_data = datasets.CIFAR10(self.root, 
                          train = False, 
                          download = True, 
                          transform = test_transforms)
    
    test_iterator = DataLoader(test_data, batch_size = self.batch_size, shuffle = False) 
    return test_iterator
