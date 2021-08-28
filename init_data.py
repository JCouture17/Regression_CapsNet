# Initialize the data

from functions import Functions
from load_images import load_images
import numpy as np

def init_data(cycles, rgb, batch_size, resize):
    import argparse
    import os
    from skimage import io
    
    dt = Functions()
    
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network for RUL Prediction")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.95, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.00005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--data_dir', default='./data',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--in_size', default=1)
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)    

    if rgb == 'y':
        args.in_size = 3
    elif rgb == 'n':
        args.in_size = 1  
        
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)   
        
    data = io.imread('Dataset_' + cycles + '.tif')
    labels = dt.load('RUL_' + cycles + '.mat', 'rul').astype(np.int16)
    testData = io.imread('Test_' + cycles + '.tif')
    testLabels = dt.load('rulTest_' + cycles + '.mat', 'rul').astype(np.int16)
    
    train_loader, test_loader = load_images(data, labels, batch_size=batch_size, resize=resize, test_size=0.2, rgb=rgb)
    test_data, test_data1 = load_images(testData, testLabels, batch_size=batch_size, resize=resize, test_size=0.01, rgb=rgb)
        
    return train_loader, test_loader, test_data, args

    
        