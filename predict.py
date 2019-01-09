

# --------------------------
# Imports
import numpy as np
import json
from PIL import Image
from math import ceil
import torch
from train import use_gpu
from train import Classifier
from torchvision import models

import argparse


# --------------------------
# Functions

# -----------
def parse_arguments():

	# Parser creation
	parser = argparse.ArgumentParser(description="Predicting Image Class")

	# Image location
	parser.add_argument('input',
		help='Path to image file as string.')

	# Model checkpoint location
	parser.add_argument('checkpoint',
		help='Path to model checkpoint file as string.')

	# Most likely classes
	parser.add_argument('--top_k',
		type=int,
		help='Number of most likely classes as int.')

	# Category to names mapper
	parser.add_argument('--category_names',
		type=str,
		help='Path to category to class mapping file as string.')

	# Enable GPU predicting
	parser.add_argument('--gpu',
		action="store_true",
		help='Enable GPU for computing (default is CPU).')

	return parser.parse_args()


# -----------
def rebuildModel(filepath):
    
    # Load model metadata
    # Loading weights for CPU model whoch were trained on GPU
    # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # Recreate the pretrained base model
    #model = models.vgg16(pretrained=True)
    model = getattr(models, checkpoint['name'])(pretrained=True)
    
    # Replace the classifier part of the model
    model.classifier = checkpoint['classifier']
    
    # Rebuild saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model


# -----------
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    
    # Find the shorter side and resize it to 256 keeping aspect ration
    # if the width > height
    if image.width > image.height:        
        # Constrain the height to be 256
        image.thumbnail((10000000, 256))
    else:
        # Constrain the width to be 256
        image.thumbnail((256, 10000000))
    
    # Center crop the image
    crop_size = 224
    left_margin = (image.width - crop_size) / 2
    bottom_margin = (image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size  
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert values to range of 0 to 1 instead of 0-255
    image = np.array(image)
    image = image / 255
    
    # Standardize values
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose(2, 0, 1)
    
    return image


# -----------
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Move model into evaluation mode and to CPU
    model.eval()
    #model.cpu()
   
    # Open image
    image = Image.open(image_path)
    
    # Process image
    image = process_image(image) 
    
    # Change numpy array type to a PyTorch tensor
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    
    # Format tensor for input into model
    # (add batch of size 1 to image)
    # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
    image = image.unsqueeze(0)
    
    # Predict top K probabilities
    # Reverse the log conversion
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(topk)
    #print(top_probs)
    #print(top_labs)
    
    # Convert from Tesors to Numpy arrays
    top_probs = top_probs.detach().numpy().tolist()[0]
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    # Map tensor indexes to classes
    labs = []
    for label in top_labs.numpy()[0]:
        labs.append(idx_to_class[label])

    return top_probs, labs


# -----------
def convertCategoryToName(categories, mapper='cat_to_name.json'):
    
    # Load json file
    with open(mapper, 'r') as f:
        cat_to_name = json.load(f)
        #print(cat_to_name)
    
        names = []

        # Find flower names corresponding to predicted categories
        for category in categories:
            names.append(cat_to_name[str(category)])

    return names


# --------------------------
def main():

	# Parsing a command-Line
	args = parse_arguments()

	# Load categories names from a json file
	print('Loading category names...')
	cat_to_name_file = 'cat_to_name.json'
	if args.category_names is not None:
		cat_to_name_file = args.category_names
	with open(cat_to_name_file, 'r') as f:
		cat_to_name = json.load(f)

	# Load model checkpoint trained with train.py
	print('Loading model definition from file...')
	model_from_file = rebuildModel(args.checkpoint)

	# Use GPU if requested
	current_device = use_gpu(model_from_file, args.gpu)

	# Predict image class
	print('Predicting image class...')
	topk = 5
	if args.top_k is not None:
		topk = 	args.top_k
	image_path = args.input
	probs, classes = predict(image_path, model_from_file, topk=topk)
	names = convertCategoryToName(classes)

	# Read the flower name based on the folder
	# number (flower class id) and external dictionary mapping
	folder_number = image_path.split('/')[2]
	label = cat_to_name[folder_number]

	print('Image location: {}'.format(image_path))
	print('Actual category number of the image: {}'.format(folder_number))
	print('Actual category name of the image: {}'.format(label))
	print('Showing {} most likely classes:'.format(args.top_k))

	index = 0
	for i, j, k in zip(probs, classes, names): 
		if j != folder_number: 
			indicator = ''
		else:
			indicator = '<--- correct prediction'
		print("Position {} {}".format(index + 1, indicator))
		print("- flower name: {}\n- class: {}\n- likelihood: {}%".format(k, j, ceil(i*100)))
		index = index + 1


# --------------------------
if __name__ == '__main__': main()

