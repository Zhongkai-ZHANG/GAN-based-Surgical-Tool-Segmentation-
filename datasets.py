import os
import cv2
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


class ImageDataSet(Dataset):
    def __init__(self, root, mode, transform):
        super(ImageDataSet, self).__init__()
        self.path = os.path.join(root, mode)
        self.image_list = [x for x in os.listdir(self.path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)  # the size of dataset

    def __getitem__(self, index):
        image_path = os.path.join(self.path, self.image_list[index])
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
        # BGR -> RGB IMREAD_COLOR : If set, always convert image to the 3 channel BGR color image.

        if self.transform is not None:
            image = self.transform(image)

        # Dataset needs to return one data and one label(the label means nothing but it can't be deleted)
        # for each data unite.
        lable = 'NONE'
        return image, lable


# Configure dataloaders
def Get_Data_Train(args):
    # data pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        transforms.Normalize(mean=(0.5019608, 0.5019608, 0.5019608), std=(0.4980392, 0.4980392, 0.4980392))  # (0, 1) -> (-1, 1)
    ])
    # Build Dataset object. mode: data set for training, test, val
    train_data = ImageDataSet(args.train_root, mode=args.case_train, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_loader


def Get_Data_Test(args):
    # data pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        transforms.Normalize(mean=(0.5019608, 0.5019608, 0.5019608), std=(0.4980392, 0.4980392, 0.4980392))  # (0, 1) -> (-1, 1)
    ])
    # Build Dataset object. mode: data set for training, test, val
    test_data = ImageDataSet(args.test_root, mode=args.case_test, transform=transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, drop_last=True)
    return test_loader

def Get_Data_Index(args):
    # data pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        transforms.Normalize(mean=(0.5019608, 0.5019608, 0.5019608), std=(0.4980392, 0.4980392, 0.4980392))  # (0, 1) -> (-1, 1)
    ])
    # Build Dataset object. mode: data set for training, test, val
    test_data = ImageDataSet(args.test_root, mode=args.case_test, transform=transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size_index, shuffle=True, drop_last=True)
    return test_loader