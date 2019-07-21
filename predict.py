# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:24:07 2018

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
from PIL import Image


#Import argument processing/commandline  module
import argparse

from train import network

def input_arg():
    """
    Set up a parser object and define the arguments to parse the command line strings in
    to relevant python data types provided.
    """
    parser = argparse.ArgumentParser(description = 'Flower Image Classifier - prediction')
    
    #Add the arguments to parse the command line
    parser.add_argument("--image_path", help="Sample Image to process",default='flowers/test/14/image_06083.jpg')
    parser.add_argument("--topk",type=int,  help="top k probabilities", default=5)
    parser.add_argument("--hidden_units", type=int, help="Number of hidden units'",default=2)
    parser.add_argument("--checkpoint_file", type=str, help="Saved model'",default='flower_classifier_model.pth')
    parser.add_argument("--class_names_path", type=str, help="Path to class_names'", default='cat_to_name.json')
    parser.add_argument('--gpu', type=bool, help="device for processing'",default=True)
    parser.add_argument("--epochs", type=int, help="Number of epochs'")
    
    args=parser.parse_args()
    
    return args
args=input_arg()

#Dictionary mapping from category labels to category flower names
with open(args.class_names_path, 'r') as f:
    cat_to_name = json.load(f)


# Write a function that loads a checkpoint and rebuild the model
def load_model(filepath):
    checkpoint = torch.load(filepath)
    hidden_sizes = checkpoint['hidden_sizes']
    pretrained_nn = checkpoint['pretrained_nn']
    output_size = checkpoint['output_size']
    model,criterion,optimizer=network(pretrained_nn,hidden_sizes,lr=.002)
    class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model,class_to_idx

model,class_to_idx=load_model(args.checkpoint_file)

## Image preprocessing
# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,returns an Numpy array
    Args:Image to be processed
    Returns : processed image
    ''' 
    img_transforms=transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])
    pil_img=Image.open(image)
    pil_img_t=img_transforms(pil_img)
    pil_img_t=np.array(pil_img_t)  #convert the tensor to a numpy
    
    
    return pil_img_t

#image=test_dir + '/14' + '/image_06083.jpg'
image=args.image_path    
image=process_image(image)


# TODO: Implement the code to predict the class from an image file
def predict(image_path, model,topk=5):
    ''' 
    Predict the class (or classes) of an image using a saved deep learning model.
    Args :image to be predicted ,model architecture, topk :top k element(s)
    Returns: Top probabilites,category label,category flower name
    '''
    image_pred=process_image(image_path)
    image_pred = torch.from_numpy(image_pred).type(torch.FloatTensor)
    device=torch.device('cuda' if args.gpu ==True and torch.cuda.is_available()  else 'cpu')
    model.to(device)
    image_pred.to(device)
    #if torch.cuda.is_available():
    #    model.to('cuda')
    #    image_pred.to('cuda')
    #else:
    #    print('Ensure cuda is available')
        
    model.eval()
    with torch.no_grad():
        image_pred.unsqueeze_(0) #adds the batch dimension.Feeding a single image to the model (or Modules in general) needs input with a batch dimension at position 0.
        output=model.forward(image_pred.to(device))
        
        #probs=F.softmax(output.data,dim=1)
        probs=torch.exp(output.to('cpu'))
        top_probs,top_classes=probs.topk(args.topk)
        top_probs = top_probs.detach().numpy().tolist()[0]
        top_classes = top_classes.detach().numpy().tolist()[0]
        
        #switch the key value pair
        idx_to_class={val:key for key,val in class_to_idx.items()}
        idx_to_class.items()
        
        #Get the flowers corresponding to the classes
        top_flowers=[cat_to_name[idx_to_class[label]] for label in top_classes]
        return top_probs,top_classes,top_flowers

#image_path=test_dir + '/14' + '/image_06083.jpg'
top_probs,top_classes,top_flowers=predict(args.image_path, model,topk=args.topk) 

print("Top Probablities: {} \nTop Category labels : {}  \nTop Category Flower names :{}".format(top_probs,top_classes,top_flowers))
print('\nImage predicted with probability of {} with the identification of category flowername: {}'.format(top_probs[0],top_flowers[0]))


