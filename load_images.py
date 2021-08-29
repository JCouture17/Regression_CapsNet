# Load images

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader

def load_images(data, targets, batch_size, resize=128, test_size=0.2, rgb = 'y'):    
    if rgb == 'y':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((resize, resize))
            ])
    elif rgb == 'n':
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Resize((resize, resize))
            ])
            
    if test_size != 0:
        x_train, x_test, y_train, y_test \
            = train_test_split(data, targets, test_size=test_size, shuffle=True)        
        # Build dataset
        train_data = CustomImageDataset(dataset=(x_train, y_train), transform=transform)
        test_data = CustomImageDataset(dataset=(x_test, y_test), transform=transform)
        # Build dataloader
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
        return train_loader, test_loader
        
    elif test_size == 0:
        train_data = CustomImageDataset(dataset=(data, targets), transform=transform)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)    
        return train_loader

class CustomImageDataset():
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, index):
        x = self.dataset[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.dataset[1][index]
        return x, y 

