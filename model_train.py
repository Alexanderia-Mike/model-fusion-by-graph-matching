'''
in this file, I tried to build an extremely simple model for MNIST training, with only
two convolutional layers but still having a relatively high accuracy. 
The methodology is obtained from the source:
    https://www.kaggle.com/lakhindr/small-efficient-neural-network-for-mnist
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

'''
device configuration
'''
# device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
device = torch.device( 'cpu' )

'''
hyperparameters
'''
input_width = 28
input_size = input_width ** 2

# activation is 'relu'
# loss is 'cross entropy loss'
# optimizer is 'sgd'
dropout = True
conv_padding = 'same'
maxpool_padding = 0
conv_kernel_size = 5
maxpool_kernel_size = 2

hidden_size_1_conv = 32
hidden_size_2_conv = 64
input_size_fc = int( hidden_size_2_conv * (input_width / maxpool_kernel_size ** 2) ** 2 )

num_classes = 10
num_epochs = 48
batch_size = 32
learning_rate = 0.00075


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
	batch_size=batch_size,
	shuffle=True )
test_loader = torch.utils.data.DataLoader( 
    dataset=test_dataset,
	batch_size=batch_size,
	shuffle=False )

'''
define the neural network class
'''
class SimpleNet( nn.Module ):
    def __init__( self ):
        super( SimpleNet, self ).__init__()
        # TODO: add dropout
        self.conv1 = nn.Conv2d( 1, hidden_size_1_conv, conv_kernel_size, padding=conv_padding, device=device )
        self.maxpool = nn.MaxPool2d( maxpool_kernel_size, padding=maxpool_padding )
        self.conv2 = nn.Conv2d( hidden_size_1_conv, hidden_size_2_conv, conv_kernel_size, padding=conv_padding, device=device )
        self.fc1 = nn.Linear( input_size_fc, num_classes, device=device )
        
    def forward( self, x ):
        output = F.relu( self.conv1( x ) )
        output = self.maxpool( output )
        output = F.relu( self.conv2( output ) )
        output = self.maxpool( output )
        output = output.view( -1, input_size_fc )
        output = self.fc1( output )
        return output

model = SimpleNet()

'''
define the loss and optimizer
'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD( model.parameters(), lr=learning_rate )

'''
training loops
'''
num_total_steps = len( train_loader )
for epoch in range( num_epochs ):
    for i, ( images, labels ) in enumerate( train_loader ):
        '''
        size: 	100, 1, 28, 28
        '''
        images = images.to( device )
        labels = labels.to( device )
        
        outputs = model( images )
        
        '''
        cross entropy loss calculation whose
	        first argument is prediction of size [num_samples] x [num_categories]
	        second argument is true label of size [num_samples] x 1
        '''
        loss = criterion( outputs, labels )
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        if ( i + 1 ) % 100 == 0:
            print( f'epoch {epoch + 1} / {num_epochs}, \
            	step {i + 1} / {num_total_steps}, \
            	loss = {loss.item():.4f}' )

'''
testing
'''
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to( device )
        labels = labels.to( device )
        
        outputs = model( images )
        
        # value, index
        _, predictions = torch.max( outputs, 1 )
        n_samples += labels.shape[0]
        n_correct += ( predictions == labels ).sum().item()
    acc = 100 * n_correct / n_samples
    print( f'accuracy = {acc}%' )