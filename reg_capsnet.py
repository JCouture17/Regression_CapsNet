'''
For use in the paper: Novel Image-Based Rapid RUL Prediction for Li-Ion Battery using Capsule Network
Author: Jonathan Couture, OntarioTech University
E-mail: jonathan.couture@ontariotechu.net
Date : August 29th 2021

    If you use this code, modified or not, please cite the aforementioned paper
'''

from capsule_layers import DenseCapsule, PrimaryCapsule
from init_data import init_data

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
import numpy as np
from time import time

class CapsuleNet(nn.Module):
    def __init__(self, input_size, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.routings = routings
        
        # For 128 x 128 pixel images
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=(5, 5), stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(5, 5), stride=2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(5, 5), stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(5, 5), stride=1)
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=3, stride=1)
        self.digitcaps = DenseCapsule(in_num_caps=32*6*6, in_dim_caps=8,
                                      out_num_caps=256, out_dim_caps=16, routings=routings)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(256*16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))  
        x = self.relu(self.conv5(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        x = self.flatten(x)
        x = self.decoder(x)
        return x

def rmse_loss(y_true, y_pred):
    se = (torch.sub(y_true, y_pred))**2
    rmse = ((torch.sum(se) / len(y_true.data))**0.5)
    return rmse

def mae_loss(y_true, y_pred):
    mae = (torch.sum(torch.abs(torch.sub(y_true, y_pred)))) / len(y_true.data)
    return mae

def test(model, test_loader):
    error, se, pe, test_loss = 0, 0, 0, 0
    model.eval()
    for x, y in test_loader:
        x, y = Variable(x.cuda()), Variable(y.cuda())
        y_pred = model(x)
        test_loss += (mae_loss(y, y_pred).detach() + rmse_loss(y, y_pred).detach()) * x.size(0)
        # Metrics
        error += torch.sum(torch.abs(torch.sub(y_pred.data, y.data)))
        se += torch.sum((torch.abs(y_pred.data - y.data))**2)
        pe += torch.sum(torch.abs((y.data - y_pred.data)/y.data))
    mape = (1 - (pe/(len(test_loader.dataset))))*100 # Mean absolute percentage error (%)
    mae = (error/(len(test_loader.dataset))) # Mean absolute error (cycles)
    rmse = (se / len(test_loader.dataset))**0.5
    test_loss /= len(test_loader.dataset)
    return test_loss, mape, mae, rmse

def train(model, train_loader, test_loader, test_data, args, epochs, early_stop=5):
    print('Begin Training' + '-'*70)

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_mae, best_rmse = 500, 500
    
    early_stopping = EarlyStopping(patience=early_stop)
    for epoch in range(epochs):
        model.train()  # set to training mode
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable
            y_pred = model(x)  # forward
            loss = mae_loss(y, y_pred) + rmse_loss(y, y_pred)  # combining the two losses
            loss.backward()
            training_loss += loss.detach() * x.size(0)
            optimizer.step()
            optimizer.zero_grad()
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        val_loss, val_acc, mae, rmse = test(model, test_loader)
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, \n val_acc=%.4f, MAE=%.4f RMSE=%.4f time=%ds"
              % (epoch+1, training_loss / len(train_loader.dataset),
                  val_loss, val_acc, mae, rmse, time() - ti))
        
        # Testing accuracy for each cells throughout the epochs
        mean_mae, mean_rmse = 0, 0
        test_loss, test_acc, test_mae, test_rmse = test(model, test_data['bat1'])
        print('cell 1 test acc = %.4f, mae = %.4f, rmse = %.4f'\
          % (test_acc, test_mae, test_rmse))
        mean_mae += test_mae; mean_rmse += test_rmse
        test_loss, test_acc, test_mae, test_rmse = test(model, test_data['bat2'])
        print('cell 2 test acc = %.4f, mae = %.4f, rmse = %.4f'\
          % (test_acc, test_mae, test_rmse))
        mean_mae += test_mae; mean_rmse += test_rmse
        test_loss, test_acc, test_mae, test_rmse = test(model, test_data['bat3'])
        print('cell 3 test acc = %.4f, mae = %.4f, rmse = %.4f'\
          % (test_acc, test_mae, test_rmse))
        mean_mae += test_mae; mean_rmse += test_rmse
        test_loss, test_acc, test_mae, test_rmse = test(model, test_data['bat4'])
        print('cell 4 test acc = %.4f, mae = %.4f, rmse = %.4f'\
          % (test_acc, test_mae, test_rmse))
        mean_mae += test_mae; mean_rmse += test_rmse
        print('Average MAE = %.4f, Average RMSE = %.4f' % (mean_mae/4, mean_rmse/4))
        if best_mae > mean_mae/4:
            best_mae = mean_mae/4
            mae_epoch = epoch + 1
        if best_rmse > mean_rmse/4:
            best_rmse = mean_rmse/4
            rmse_epoch = epoch + 1
        print('best MAE is epoch %.2d and best RMSE is epoch %.2d' % (mae_epoch, rmse_epoch))
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl') # Use this to save the model to a .pkl file
    # print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model 
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./result/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


if __name__ == "__main__":
    ''' These first parameters are user defined:
        - resize: parameter that dictates the input size of the images into the neural network. 
            NOTE: the size of the convolutional filters would need to be changed if the input size is altered.
        - cycles: available datasets cover the following inputs ('1', '3', '5', and '10'). 
        - rgb: whether or not to use 3 seperate input channels for the colors, 
            or if the input should be transformed into grayscale images. ('y', 'n')
        - transfer: whether or not to load in a network's state_space to make use of transfer learning ('y', 'n')
    '''
    
    resize = 128
    batch_size = 24
    cycles = '1'
    rgb = 'y'
    transfer = 'y'
    early_stop = 2
    lr = 1E-2
    
    train_data, val_data, testingData, args = init_data(cycles, rgb, batch_size, resize)  # Intializes the train/val/test sets
    args.lr = lr  # Redefining the learning rate with the previous user defined
    
    # Creating the Capsule Network
    from torchinfo import summary
    model = CapsuleNet(input_size=[args.in_size, resize, resize], routings=3)
    
    # Transfer Learning
    if transfer == 'y':
        model.load_state_dict(torch.load('./result/trained_model_39.pkl'))
        for param in model.parameters():  # Deactivate training for all layers
            param.requires_grad = False
        decoder = nn.Sequential(  # Replace the final sequential layers
            nn.Dropout(0.3), 
            nn.Linear(256*16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
            )
        model.decoder = decoder
    
    model.cuda()  # Put model on GPU
    print(summary(model, (batch_size, args.in_size, 128, 128)))  # Print the summary of the architecture

    # Training and Testing
    train(model, train_data, val_data, testingData, args, epochs=100, early_stop=early_stop)
