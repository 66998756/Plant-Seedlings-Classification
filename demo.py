import os
import argparse
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from model import ResNeXt
from dataloader import remove_background

parser = argparse.ArgumentParser()
parser.add_argument('--id', default='./checkpoint/', help='id number of training process')
parser.add_argument('--target_epoch', '--epoch', default=110, help='checkpoint epoch')
parser.add_argument('--weight_path', '--w', default='./checkpoint/', help='weight path')
parser.add_argument('--img_resize', type=int, help='resized img size')
parser.add_argument('--device', type=int, help='target GPU')
opt = parser.parse_args()

img_label_dict = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed",
                  "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize", 
                  "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
                  "Sugar beet"]

def get_setting(log_id):
    with open('./figure/log_{}.txt'.format(log_id), 'r') as log_file:
        lines = []
        for line in log_file:
            lines.append(line)
        
        logs = {
            "id": lines[2][lines[2].find(' '):-1],
            "input_img_resize": lines[3][lines[3].find(' '):-1],
            "epochs": lines[4][lines[4].find(' '):-1],
            "lr": lines[5][lines[5].find(' '):-1],
            "batch_size": lines[6][lines[6].find(' '):-1]
        }

        return logs


class testDataset(Dataset):
    def __init__(self, data_path, img_resize=192, transform=None):
        self.test_transform = transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize([img_resize, img_resize]), 
                                 # remove_background(),
                                 transforms.ToTensor(), 
                                 # transforms.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500]),
                             ])
        self.data_path = data_path
        self.img_path = os.listdir(data_path)
    
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_path, self.img_path[item])
        img_name = self.img_path[item]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.test_transform(img)

        return img, img_name


def demo():
    config = get_setting(opt.id)
    configs = {"num_blocks":[3, 4, 23, 3], "cardinality":64, "bottleneck_width":4}
    
    with torch.cuda.device(opt.device):

        model = ResNeXt(num_blocks=configs["num_blocks"], cardinality=configs["cardinality"], bottleneck_width=configs["bottleneck_width"], num_classes=12, input_size=[int(config['input_img_resize']), int(config['input_img_resize'])]).to(opt.device)
        checkpoint = torch.load('./checkpoint/model_{}_{}.pt'.format(opt.id, opt.target_epoch), map_location=torch.device('cuda:{}'.format(opt.device)))
        model.load_state_dict(checkpoint['model_state_dict'])

        # model = torch.load('./checkpoint/model_{}_{}.pt'.format(opt.id, opt.target_epoch), map_location=torch.device('cuda:{}'.format(opt.device)))
        model.eval()
        # print(model)
        test_dataset = testDataset(data_path='./dataset/test', img_resize=opt.img_resize)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )
        
        output = {'image': [], 'result': []}
        with torch.no_grad():
            with open('output_{}.csv'.format(opt.id), 'w') as f:
                f.writelines("file,species\n")
                for img, img_name in test_dataloader:
                    img = img.to(opt.device)
                    result = model(img)
                    f.writelines("{},{}\n".format(img_name[0], img_label_dict[result.argmax(dim=1)]))
                

if __name__ == '__main__':
    demo()
    