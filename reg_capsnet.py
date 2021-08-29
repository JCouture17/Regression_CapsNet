'''
For use in the paper: Novel Image-Based Rapid RUL Prediction for Li-Ion Battery using Capsule Network
Author: Jonathan Couture, OntarioTech University
email: jonathan.couture@ontariotechu.net

    If you use this code for publishing, modified or not, please cite the abovementionned paper
'''

from CapsuleLayers import DenseCapsule, PrimaryCapsule
from init_data import init_data

from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
import numpy as np
import torch
from time import time
import csv

class CapsuleNet(nn.Module):
    def __init__(self, input_size, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.routings = routings
        
        # For 128 x 128 pixel images
        self.conv1 = nn.Conv2d(input_size[0], 258, 5, 2, groups = input_size[0])
        self.conv2 = nn.Conv2d(258, 256, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, stride=1)
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
            # nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x, y=None):
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
    rmse = (((torch.sum(se) / len(y_true.data)))**0.5)
    return rmse

def mae_loss(y_true, y_pred):
    mae = (torch.sum(torch.abs(torch.sub(y_true, y_pred)))) / len(y_true.data)
    return mae

def test(model, test_loader, args):
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
    logfile = open(args.save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc, best_mae = 0, 500
    
    early_stopping = EarlyStopping(patience=early_stop)
    for epoch in range(epochs):
        model.train()  # set to training mode
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable
            y_pred = model(x)  # forward
            loss = mae_loss(y, y_pred) + rmse_loss(y, y_pred) # Combining the two losses to obtain an optimal balance
            loss.backward()
            training_loss += loss.detach() * x.size(0)
            optimizer.step()
            optimizer.zero_grad()
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        val_loss, val_acc, mae, rmse = test(model, test_loader, args)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, \n val_acc=%.4f, MAE=%.4f RMSE=%.4f time=%ds"
              % (epoch+1, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, mae, rmse, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
        
            ## Testing accuracy for each cells throughout the epochs
        mean_mae, mean_rmse = 0, 0
        test_loss, test_acc, test_mae, test_rmse = test(model, test_data['bat1'], args)
        print('cell 1 test acc = %.4f, mean absolute error = %.4f, rmse = %.4f'\
          % (test_acc, test_mae, test_rmse))
        mean_mae += test_mae; mean_rmse += test_rmse
        test_loss, test_acc, test_mae, test_rmse = test(model, test_data['bat2'], args)
        print('cell 2 test acc = %.4f, mean absolute error = %.4f, rmse = %.4f'\
          % (test_acc, test_mae, test_rmse))
        mean_mae += test_mae; mean_rmse += test_rmse
        test_loss, test_acc, test_mae, test_rmse = test(model, test_data['bat3'], args)
        print('cell 3 test acc = %.4f, mean absolute error = %.4f, rmse = %.4f'\
          % (test_acc, test_mae, test_rmse))
        mean_mae += test_mae; mean_rmse += test_rmse
        test_loss, test_acc, test_mae, test_rmse = test(model, test_data['bat4'], args)
        print('cell 4 test acc = %.4f, mean absolute error = %.4f, rmse = %.4f'\
          % (test_acc, test_mae, test_rmse))
        mean_mae += test_mae; mean_rmse += test_rmse
        print('Average MAE = %.4f, Average RMSE = %.4f' % (mean_mae, mean_rmse))

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    logfile.close()
    # torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl') # Use this to save the model to a .pkl file
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
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
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    
if __name__ == "__main__":
    ### User Defined ###
    resize = 128
    batch_size = 4
    cycles = '1'
    rgb = 'n' 
    early_stop = 3
    lr = 0.00005

    train_data, val_data, testingData, args = init_data(cycles, rgb, batch_size, resize)
    args.lr = lr
    
    # Creating the Capsule Network
    from torchinfo import summary
    model = CapsuleNet(input_size=[args.in_size, resize, resize], routings=3)
    
    #     # Transfer Learning ##
    # model.load_state_dict(torch.load('./result/trained_model.pkl'))
    # for param in model.parameters(): # Deactivate training for all layers
    #     param.requires_grad = False
    # decoder = nn.Sequential(
    #     nn.Dropout(0.3), 
    #     nn.Linear(256*16, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 1),
    #     nn.ReLU()
    #     )
    # model.decoder = decoder
    
    model.cuda() # Put model on GPU
    print(summary(model, (batch_size, args.in_size, 128, 128))) # Print the summary of the neural network's architecture

    # Training and Testing
    train(model, train_data, val_data, testingData, args, epochs=1000, early_stop=early_stop)

