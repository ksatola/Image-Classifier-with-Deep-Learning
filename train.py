

# --------------------------
# Imports

import time
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

import argparse

# --------------------------
# Data folders
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Dataset controls
image_size = 224 # Image size in pixels
reduction = 255 # Image reduction to smaller edge 
norm_means = [0.485, 0.456, 0.406] # Normalized means of the images
norm_std = [0.229, 0.224, 0.225] # Normalized standard deviations of the images
rotation = 45 # Range of degrees for rotation
batch_size = 64 # Number of images used in a single pass
shuffle = True # Randomize image selection for a batch

# Argparse configuration
supported_architectures = ['vgg16', 'vgg19']


# --------------------------
# Functions & Classes

# -----------
def parse_arguments():

	# Parser creation
	parser = argparse.ArgumentParser(description="Training Image Classifier Settings")

	# Data folder
	parser.add_argument('data_dir',
		help='Main directory for data set as string (default is \'flowers\').')

	# Architecture selection
	parser.add_argument('--arch',
		type=str,
		help='Architecture type from tourchvision.models as string (vgg16 or vgg19).')

	# Checkpoint directory
	parser.add_argument('--save_dir',
		type=str,
		help='A folder name where the model will be saved (default is current directory).')

	# Hyperparameters
	parser.add_argument('--learning_rate',
		type=float,
		help='Gradient descent learning rate as string (default is 0.001).')
	parser.add_argument('--hidden_units',
		type=int,
		help='Number of hidden units for the input classifier layer as int.')
	parser.add_argument('--epochs',
		type=int,
		help='Number of epochs for training as int (default is 5).')

	# Enable GPU training
	parser.add_argument('--gpu',
		action="store_true",
		help='Enable GPU for computing (default is CPU).')

	return parser.parse_args()


# -----------
def get_data(datadir):

	if datadir is None:
		datadir = data_dir

	# Create transforms pipelines to run/apply them in sequence on image data
	# Next convert image data to sensors and normalize it to make backpropagation more stable
	train_transforms = transforms.Compose([transforms.RandomResizedCrop(image_size),
	                                       transforms.RandomRotation(rotation),
	                                       transforms.RandomHorizontalFlip(),
	                                       transforms.RandomVerticalFlip(),
	                                       transforms.ToTensor(),
	                                       transforms.Normalize(norm_means, norm_std)])

	valid_transforms = transforms.Compose([transforms.Resize(reduction),
	                                      transforms.CenterCrop(image_size),
	                                      transforms.ToTensor(),
	                                      transforms.Normalize(norm_means, norm_std)])

	test_transforms = transforms.Compose([transforms.Resize(reduction),
	                                      transforms.CenterCrop(image_size),
	                                      transforms.ToTensor(),
	                                      transforms.Normalize(norm_means, norm_std)])

	# Load and transform image data
	train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
	valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
	test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

	# Using the image datasets and the transforms, define the dataloaders
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
	validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

	return trainloader, validloader, testloader, train_data


# -----------
def initialize_model(arch="vgg16"):

	if arch not in supported_architectures:
		arch_ = supported_architectures[0]
	else:
		arch_ = arch

	# Load a pre-trained network
	# https://pytorch.org/docs/stable/torchvision/models.html
	# VGG16
	#model = models.vgg16(pretrained=True)
	#print('Using {} architecture.'.format(arch_))
	model = getattr(models, arch_)(pretrained=True)
	model.name = arch_

	# Freeze model parameters so we don't backprop through them
	for param in model.parameters():
		param.requires_grad = False

	return model
		

# -----------
# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
class Classifier(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.2):
        super().__init__()
        
        # Add input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add hidden layers
        h_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h_input, h_output) for h_input, h_output in h_layers])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Dropout module with drop_out drop probability
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        # Flaten tensor input
        x = x.view(x.shape[0], -1)

        # Add dropout to hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))        

        # Output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x

	
# -----------
def build_classifier(hidden_units=4096):

	input_size = 25088
	output_size = 102
	if hidden_units is None:
		hidden_units = 4096
	hidden_layers = [hidden_units, 1024]
	#hidden_layers.append(int(hidden_units))
	#hidden_layers.append(1024)
	drop_out = 0.2

	return Classifier(input_size, output_size, hidden_layers, drop_out)


# -----------
def initialize_training():

	# Define the loss function
	criterion = nn.NLLLoss()

	# Hyperparameters
	drop_out = 0.2
	# Optimizer type and learning rate
	learningearning_rate = 0.003
	# Define weights optimizer (backpropagation with gradient descent)
	# Only train the classifier parameters, feature parameters are frozen
	optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


# -----------
# A function used for validation and testing
def testClassifier(model, criterion, testloader, current_device):
    
    # Move the network and data to current hardware config (GPU or CPU)
    model.to(current_device)
        
    test_loss = 0
    accuracy = 0
        
    # Looping through images, get a batch size of images on each loop
    for inputs, labels in testloader:

        # Move input and label tensors to the default device
        inputs, labels = inputs.to(current_device), labels.to(current_device)

        # Forward pass, then backward pass, then update weights
        log_ps = model.forward(inputs)
        batch_loss = criterion(log_ps, labels)
        test_loss += batch_loss.item()

        # Convert to softmax distribution
        ps = torch.exp(log_ps)
        
        # Compare highest prob predicted class with labels
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        # Calculate accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return test_loss, accuracy


# -----------
# A function used for training (and tests with different model hyperparameters)
def trainClassifier(model, epochs_no, criterion, optimizer, trainloader, validloader, current_device):
    
    # Move the network and data to current hardware config (GPU or CPU)
    model.to(current_device)
    
    epochs = epochs_no
    steps = 0
    print_every = 1
    running_loss = 0

    # Looping through epochs, each epoch is a full pass through the network
    for epoch in range(epochs):
        
        # Switch to the train mode
        model.train()

        # Looping through images, get a batch size of images on each loop
        for inputs, labels in trainloader:

            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(current_device), labels.to(current_device)

            # Clear the gradients, so they do not accumulate
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Track the loss and accuracy on the validation set to determine the best hyperparameters
        if steps % print_every == 0:

            # Put in evaluation mode
            model.eval()

            # Turn off gradients for validation, save memory and computations
            with torch.no_grad():

                # Validate model
                test_loss, accuracy = testClassifier(model, criterion, validloader, current_device)
                
            train_loss = running_loss/print_every
            valid_loss = test_loss/len(validloader)
            valid_accuracy = accuracy/len(validloader)

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss:.3f}.. "
                  f"Test loss: {valid_loss:.3f}.. "
                  f"Test accuracy: {valid_accuracy:.3f}")

            running_loss = 0
            
            # Switch back to the train mode
            model.train()
                
    # Return last metrics
    return train_loss, valid_loss, valid_accuracy


# -----------
def saveCheckpoint(model, train_data, savedir=''):
    
    # Mapping of classes to indices
    model.class_to_idx = train_data.class_to_idx
    
    # Create model metadata dictionary
    checkpoint = {
        'name': model.name,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict()
    }

    # Save to a file
    timestr = time.strftime("%Y%m%d_%H%M%S")
    file_name = 'model_' + timestr + '.pth'

    if not savedir is None:
    	if not os.path.exists(savedir):
    		os.makedirs(savedir)
    	file_name = os.path.join(savedir, file_name)

    torch.save(checkpoint, file_name)
    
    return file_name


# -----------
def use_gpu(model, gpu):

	#print(gpu)

	# Check computer hardware
	current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#print(current_device)
	if current_device == 'cpu':
		print('Your current device supports only CPU.')
		print('Using CPU for processing.')
	else:
		print('Your current device supports GPU/CUDA.')
		if gpu:
			print('Using GPU for processing.')
		else:
			current_device = 'cpu'
			print('Using CPU for computing as GPU not requested. Use \'--gpu\' switch to leverage GPU/CUDA.')

	# Send model to device
	model.to(current_device)

	return current_device


# --------------------------
def main():

	# Parsing a command-Line
	args = parse_arguments()

	trainloader, validloader, testloader, train_data = get_data(args.data_dir)

	model = initialize_model(args.arch)
	model.classifier = build_classifier(args.hidden_units)
	print('Model architecture: \n{}'.format(model))

	current_device = use_gpu(model, args.gpu)

	# ------
	# Start training

	print('Initializing...')

	# Define the loss function
	criterion = nn.NLLLoss()

	# Hyperparameters
	drop_out = 0.2

	learning_rate = args.learning_rate
	if args.learning_rate is None:
		learning_rate = 0.003

	optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

	epochs_no = args.epochs
	if args.epochs is None:
		epochs_no = 5

	print('Training started...')
	print('Be patient. Depending on if you use CPU or GPU, it may take some (longer) time until first epoch passes.')

    # Train and validate the neural network classifier
	train_loss, valid_loss, valid_accuracy = trainClassifier(model, 
		epochs_no, criterion, optimizer, trainloader, validloader, current_device)

	# Display final summary
	print("Final result \n",
	      f"Train loss: {train_loss:.3f}.. \n",
	      f"Test loss: {valid_loss:.3f}.. \n",
	      f"Test accuracy: {valid_accuracy:.3f}")

	print('Training complete.')


	# Save model to a file
	filename = saveCheckpoint(model, train_data, args.save_dir)

	print('Model saved to: {}.'.format(filename))
			

# --------------------------
if __name__ == '__main__': main()

