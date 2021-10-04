'''
in this file, I tried to build an extremely simple model1 for MNIST training, with only
two convolutional layers but still having a relatively high accuracy. 
The methodology is obtained from the source:
    https://www.kaggle.com/lakhindr/small-efficient-neural-network-for-mnist
'''

from numpy import dot, mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch.serialization import load
import torchvision
import torchvision.transforms as transforms
import time

import model_gm
import baseline_gm
import utils_gm
import wasserstein_ensemble_gm
import fusion_gm

# training_mode = 'same'
# training_mode = 'iid'
training_mode = 'niid'

'''
device configuration
'''
# device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
device = torch.device( 'cpu' )

'''
hyperparameters
'''
args = utils_gm.dotdict( {
    "device": device,
    "model_name": "simplemnistnet",

    # hyperparameters
    "input_width": 28,
    "input_size": 28 ** 2,

    "conv_padding": 'same',
    "maxpool_padding": 0,
    "conv_kernel_size": 5,
    "maxpool_kernel_size": 2,

    "hidden_size_1_conv": 32,
    "hidden_size_2_conv": 64,
    "input_size_fc": int( 64 * (28 / 2 ** 2) ** 2 ),

    "num_classes": 10,
    "num_epochs": 48,
    "batch_size": 32,
    "learning_rate": 0.00075,
    "bias": False,

    # for ensembling
    "activation_histograms": True,
    "act_num_samples": 200,
    "clip_gm": False,
    "clip_max": 5,
    "clip_min": 0,
    "correction": True,
    "dataset": "mnist",
    "debug": False,
    "dist_normalize": True,
    "ensemble_step": 0.5,
    "eval_aligned": False,
    "exact": True,
    "geom_ensemble_type": "wts",
    "gpu_id": -1,
    "ground_metric": "euclidean",
    "ground_metric_eff": False,
    "ground_metric_normalize": "none",
    "importance": None,
    "normalize_wts": False,
    "num_models": 2,
    "not_squared": True,
    "past_correction": True,
    "prediction_wts": True,
    "proper_marginals": False,
    "reg": 0.01,
    "skip_last_layer": False,
    "softmax_temperature": 1,
    "unbalanced": False,
    "weight": [0.5, 0.5],
    "width_ratio": 1
} )

SimpleNet = model_gm.SimpleNet

# activation is 'relu'
# loss is 'cross entropy loss'
# optimizer is 'sgd'

'''
load MNIST dataset
'''
train_dataset = torchvision.datasets.MNIST( 
    root='./data', 							# the directory to store the dataset
    train=True, 							# the dataset is used to train
    transform=transforms.ToTensor(), 		# the dataset is in the form of tensors
    download=True )							# if the dataset is directly available, download 
test_dataset = torchvision.datasets.MNIST( 
    root='./data', 							# the directory to store the dataset
    train=False, 							# the dataset is used to test
    transform=transforms.ToTensor(), 		# the dataset is in the form of tensors
    download=False )
train_loader = torch.utils.data.DataLoader( 
    dataset=train_dataset,
	batch_size=args.batch_size,
	shuffle=True )
test_loader = torch.utils.data.DataLoader( 
    dataset=test_dataset,
	batch_size=args.batch_size,
	shuffle=False )

'''
define the neural network class
'''

model1 = model_gm.get_model_from_name( args )
model2 = model_gm.get_model_from_name( args )

'''
define the loss and optimizer
'''
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD( model1.parameters(), lr=args.learning_rate )
optimizer2 = torch.optim.SGD( model2.parameters(), lr=args.learning_rate )

'''
training loops for model1
'''
num_total_steps = len( train_loader )
for epoch in range( args.num_epochs ):
    for i, ( images, labels ) in enumerate( train_loader ):
        if training_mode == 'same':
            pass
        elif training_mode == 'iid':
            if i % 2 == 0:
                continue
        elif training_mode == 'niid':
            # only select the digits whose values are neither 0 nor 1
            mask = ( (labels != 0).to(torch.int) * (labels != 1).to(torch.int) ).to(torch.bool)
            labels = labels[mask]
            images = images[mask]
        else:    
            raise NotImplementedError
        '''
        size: 	100, 1, 28, 28
        '''
        images = images.to( device )
        labels = labels.to( device )
        
        outputs1 = model1( images )
        
        '''
        cross entropy loss calculation whose
	        first argument is prediction of size [num_samples] x [num_categories]
	        second argument is true label of size [num_samples] x 1
        '''
        loss = criterion( outputs1, labels )
        
        optimizer1.zero_grad()
        loss.backward()
        
        optimizer1.step()
        
        if ( i + 1 ) % 100 == 0:
            print( f'epoch {epoch + 1} / {args.num_epochs}, \
            	step {i + 1} / {num_total_steps}, \
            	loss = {loss.item():.4f}' )

'''
training loops for model2
'''
num_total_steps = len( train_loader )
for epoch in range( args.num_epochs ):
    for i, ( images, labels ) in enumerate( train_loader ):
        if training_mode == 'same':
            pass
        elif training_mode == 'iid':
            if i % 2 == 1:
                continue
        elif training_mode == 'niid':
            # only select the digits whose values are neither 8 nor 9
            mask = ( (labels != 8).to(torch.int) * (labels != 9).to(torch.int) ).to(torch.bool)
            labels = labels[mask]
            images = images[mask]
        else:    
            raise NotImplementedError
        '''
        size: 	100, 1, 28, 28
        '''
        images = images.to( device )
        labels = labels.to( device )
        
        outputs2 = model2( images )
        
        '''
        cross entropy loss calculation whose
	        first argument is prediction of size [num_samples] x [num_categories]
	        second argument is true label of size [num_samples] x 1
        '''
        loss = criterion( outputs2, labels )
        
        optimizer2.zero_grad()
        loss.backward()
        
        optimizer2.step()
        
        if ( i + 1 ) % 100 == 0:
            print( f'epoch {epoch + 1} / {args.num_epochs}, \
            	step {i + 1} / {num_total_steps}, \
            	loss = {loss.item():.4f}' )

'''
save the model1 and model2
'''
torch.save( model1, './saved_models/simplenet1_niid.pt' )
torch.save( model2, './saved_models/simplenet2_niid.pt' )

'''
load the model1 and model2
'''
# model1 = torch.load( './saved_models/simplenet1.pt' )
# model2 = torch.load( './saved_models/simplenet2.pt' )

'''
testing for models
'''
with torch.no_grad():
    n_correct1 = 0
    n_correct2 = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to( device )
        labels = labels.to( device )
        
        outputs1 = model1( images )
        outputs2 = model2( images )
        
        # value, index
        _, predictions1 = torch.max( outputs1, 1 )
        _, predictions2 = torch.max( outputs2, 1 )
        n_samples += labels.shape[0]
        n_correct1 += ( predictions1 == labels ).sum().item()
        n_correct2 += ( predictions2 == labels ).sum().item()
    acc1 = 100 * n_correct1 / n_samples
    acc2 = 100 * n_correct2 / n_samples
    print( f'model1 accuracy = {acc1}%, model2 accuracy = {acc2}%' )


''' ############################################################## for debug:
loaded_model1 = torch.load( './saved_models/simplenet.pt' )

print( f'model1 is \n{model1}\nand loaded_model is {loaded_model}' )

for (name, parameter1), parameter2 in zip( model1.named_parameters(), loaded_model.parameters() ):
    if torch.equal(parameter1, parameter2):
        print( f'for name {name}, two parameters are equal' )
    else:
        print( f'for name {name}, two parameters are different!!!' )
    ############################################################## debug end
''' 

'''
comparing
    1. naive ensembling
    2. prediction ensembling
    3. OT based fusion
    4. gm based fusion,
based on different ensembling steps
'''

for step in range( 11 ):
    ensemble_step = step / 10
    print( f'##########################################\nweight is [{ensemble_step}, {1-ensemble_step}]\n' )
    args.ensemble_step = ensemble_step

    print("------- Naive ensembling of weights -------")
    naive_acc, naive_model = baseline_gm.naive_ensembling(args, [model1, model2], test_loader)

    print("------- Prediction based ensembling -------")
    prediction_acc = baseline_gm.prediction_ensembling(args, [model1, model2], test_loader)

    print("------- OT based Ensembling on weights -------")
    geometric_acc, geometric_model = \
        wasserstein_ensemble_gm.geometric_ensembling_modularized(
        args, [model1, model2], train_loader, test_loader)

    print("------- gm based Ensembling -------")
    st_time = time.perf_counter()
    fused_model, completeness = fusion_gm.get_fused_model( args, [model1, model2] )

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = fused_model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print( f'completeness is {completeness}' )

    end_time = time.perf_counter()
    print( f'time consumed: {end_time-st_time}' )