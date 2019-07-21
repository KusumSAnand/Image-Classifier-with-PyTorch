# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:15:46 2018

@author: Kusum
"""

# Import the required libraries

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import json

#import helper 
from torchvision import datasets, transforms,models
from collections import OrderedDict

from PIL import Image
import os

#Import argument processing/commandline  module
import argparse


def input_arg():
    """
    Set up a parser object and define the arguments to parse the command line strings in
    to relevant python data types.
 
    """
    parser = argparse.ArgumentParser(description = 'Flower Image Classifier - Train a neural network and save the model')
    
    #Add the arguments to parse the command line
    parser.add_argument("--arch", help="Consider densenet or vgg as the model architecture",default='densenet')
    parser.add_argument("--lr", type=float, help="Learning rate'", default=1e-2)
    parser.add_argument("--hidden_units", type=int, help="Number of hidden units'",default=[500,300])
    parser.add_argument("--epochs", type=int, help="Number of epochs'")
    parser.add_argument("--data_dir", type=str, help="Path to dataset'",default='flowers')
    parser.add_argument("--gpu", type=bool, help="device for processing'",default=True)
    parser.add_argument("--class_names_path", type=str, help="Path to class_names'", default='cat_to_name.json')
    parser.add_argument("--checkpoint_file", type=str, help="Saved model'")
    parser.add_argument("--topk",type=int,  help="top k probabilities", default=5)
    
    args=parser.parse_args()
    #print (args)
    return args

args =input_arg()
#print(args)

def data_prep(args):
    """
    The function defines parameters to transform the image files from the path .
    The transformed images are loaded with Imagefolder and these image datasets are loaded into dataloaders
    to be processed by neural network                                                                   
    input - main folder location for the images
    output - dataloaders and imagefolders
    """
    data_dir = args
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_data_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    validation_data_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir,transform=train_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir,transform=test_data_transforms)
    validation_image_datasets = datasets.ImageFolder(valid_dir,transform=validation_data_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets,batch_size=64,shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets,batch_size=32,shuffle=True)
    validation_dataloaders = torch.utils.data.DataLoader(validation_image_datasets,batch_size=32,shuffle=True)
    return train_dataloaders,validation_dataloaders,test_dataloaders,train_image_datasets

train_dataloaders,validation_dataloaders,test_dataloaders,train_image_datasets = data_prep(args.data_dir)

    
#Dictionary mapping from category labels to category flower names
with open(args.class_names_path, 'r') as f:
    cat_to_name = json.load(f)

        
## Build and train your network
def network(pretrained_nn,hidden_sizes=args.hidden_units,lr=args.lr,output_size=102):
    '''
    Model selection and its input size
    Build the neural network for the model selected as a classifier by freezing the features. 
    define the loss and the optimizer.
    Input: the desired model selecting between densenet and vgg,hidden sizes,learning rate,output size 
    Output: model archtecture and its input size,criterion,optimizer 
    '''
    #pretrained_nn=args
    if pretrained_nn=='densenet':
        model=models.densenet121(pretrained=True)
        input_size=1024
   
    elif pretrained_nn=='vgg':
        model=models.vgg19(pretrained=True)
        input_size=25088
    else:
        print('select either vgg or densenet to build the model')
       
    #Model parametrs will be frozen so as to not backprogate through them
    for param in model.parameters():
        param.requires_grad=False
    
    #Build feed-forward network using ReLU activations and dropout
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('dropout_1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('dropout_2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(hidden_sizes[1],output_size)),
        ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier
    
    #Define the loss
    criterion = nn.NLLLoss()

    #Define the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model,criterion,optimizer

#hidden_sizes=[500,300]

model,criterion,optimizer=network(args.arch,args.hidden_units,args.lr)


#Track the loss and accuracy on the validation set to determine the best hyperparameters'
# Implement a function for the validation pass
def validation(model, validation_dataloaders, criterion):
    validation_loss = 0
    accuracy = 0
    print('Validation process in progress')
   
# Use GPU for processing if GPU is available and is required via command line argument.
    device=torch.device('cuda' if args.gpu ==True & torch.cuda.is_available()  else 'cpu')
    model.to(device)
  
    for images, labels in validation_dataloaders:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model.forward(images)
        validation_loss += criterion(output, labels).item()
        
        #Accuracy calculation
        #Find the exponential of the output (in log softmax) to get the probabilities
        ps = torch.exp(output).data
        #Class with highest probability is the predicted class and compared to true labels of the data
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validation_loss, accuracy

#validation_loss, accuracy=validation(model, validation_dataloaders, criterion)
        
#Train the classifier layers using backpropagation using the pre-trained network to get the features
def train_deep_learning(epochs, model, trainloader,print_every, criterion, optimizer):
    epochs = epochs
    print_every = print_every
    steps = 0
    device=torch.device('cuda:0' if args.gpu ==True & torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('Deep learning process begins')
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # model in eval() mode for inference
                model.eval()
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validation_dataloaders, criterion)
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.4f}".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_dataloaders)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validation_dataloaders)))
                    running_loss = 0
                    
                    #Turn on train mode
                    model.train()
                
train_deep_learning(args.epochs, model, train_dataloaders, 40, criterion, optimizer) 


#Testing your network

# Do validation on the test set
def check_accuracy_on_test(testloader):    
    correct = 0
    total = 0
    model.eval()
    device=torch.device('cuda:0' if torch.cuda.is_available() & args.gpu else 'cpu')
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
    

check_accuracy_on_test(test_dataloaders)               


#  Save the checkpoint 
model.class_to_idx = train_image_datasets.class_to_idx
model.cpu



checkpoint = {'pretrained_nn':'densenet',
              'input_size': 1024,
              'hidden_sizes':[500,300],
              'output_size': 102,
              'state_dict': model.state_dict(),
              'optimizer_dict':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epochs': args.epochs}
#torch.save(checkpoint, 'flower_classifier_model.pth')
torch.save(checkpoint, args.checkpoint_file)

print("The state dict keys: \n\n", model.state_dict().keys())