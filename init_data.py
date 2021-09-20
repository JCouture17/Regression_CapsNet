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

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)   
        
    if rgb == 'y':
        args.in_size = 3
    elif rgb == 'n':
        args.in_size = 1 
        
    data = io.imread('./Datasets/Data_' + cycles + '.tif')
    labels = dt.load('./Datasets/rul_' + cycles + '.mat', 'rul').astype(np.int16)
    testData = io.imread('./Datasets/Test_' + cycles + '.tif')
    testLabels = dt.load('./Datasets/rulTest_' + cycles + '.mat', 'rul').astype(np.int16)
    
    if cycles == '1':
        bat1_end = 1152
        bat2_end = 1945
        bat3_end = 2728
    elif cycles == '3':
        bat1_end = 1148
        bat2_end = 1937
        bat3_end = 2716
    elif cycles == '5':
        bat1_end = 1144
        bat2_end = 1929
        bat3_end = 2704
    elif cycles== '10':
        bat1_end = 1134
        bat2_end = 1909
        bat3_end = 2674

    train_loader, test_loader = load_images(data, labels, batch_size=batch_size, resize=resize, test_size=0.2, rgb=rgb)
    
    bat1 = load_images(testData[1:bat1_end], testLabels[1:bat1_end], batch_size=batch_size,\
                             resize=resize, test_size = 0, rgb=rgb)
    bat2 = load_images(testData[bat1_end + 1:bat2_end], testLabels[bat1_end + 1:bat2_end], batch_size=batch_size,\
                             resize=resize, test_size = 0, rgb=rgb)
    bat3 = load_images(testData[bat2_end + 1:bat3_end], testLabels[bat2_end + 1:bat3_end], batch_size=batch_size,\
                             resize=resize, test_size = 0, rgb=rgb)
    bat4 = load_images(testData[bat3_end + 1:-1], testLabels[bat3_end + 1:-1], batch_size=batch_size,\
                             resize=resize, test_size = 0, rgb=rgb)
    
    testingData = {'bat1':bat1, 'bat2':bat2, 'bat3':bat3, 'bat4':bat4}

    return train_loader, test_loader, testingData, args

    
        