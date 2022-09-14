import os
import time
# import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torchsummary import summary

from model import ResNeXt
from dataloader import PlantDataset
from train import train, test
from opt import arg_parse

opt = arg_parse()
# os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)

def get_model_info(model, input_size=[128, 128], verbose=0):
    # x = torch.randn(opt.batch_size, 3, input_size[0], input_size[1]).to(opt.device)
    # y = model(x)
    return str(summary(model.cuda(), (3, input_size[0], input_size[1]), verbose=verbose))


def plot_loss_img(epoch_log, train_loss_log, test_loss_log, id_start_time):
    plt.plot(epoch_log, train_loss_log, label="train loss")
    plt.plot(epoch_log, test_loss_log, label="validation loss")
    plt.title("training loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./figure/{}_training_loss.jpg".format(int(id_start_time) % 100000))
    plt.cla()


def plot_acc_img(epoch_log, train_acc_log, test_acc_log, id_start_time):
    plt.plot(epoch_log, train_acc_log, label="train acc")
    plt.plot(epoch_log, test_acc_log, label="validation acc")
    plt.title("training accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("./figure/{}_training_accuracy.jpg".format(int(id_start_time) % 100000))
    plt.cla()
    

def plot_uncorrect_count_img(epoch, uncorrect_count, id_start_time):
    img_label_dict = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed",
                  "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize", 
                  "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
                  "Sugar beet"]
    plt.bar(img_label_dict, uncorrect_count, label="uncorrect number")
    plt.xticks(rotation=90)
    plt.title("{} uncorrect count".format(epoch))
    plt.xlabel("class")
    plt.ylabel("count")
    plt.legend()
    plt.savefig("./figure/{}_testing_uncorrect_count.jpg".format(int(id_start_time) % 100000))
    plt.cla()


def main():
    configs = {"num_blocks":[3, 4, 23, 3], "cardinality":64, "bottleneck_width":4}
    model = ResNeXt(num_blocks=configs["num_blocks"], cardinality=configs["cardinality"], bottleneck_width=configs["bottleneck_width"], num_classes=12, input_size=[opt.img_size, opt.img_size]).to(opt.device)
    # optimizer = optim.Adam()
    # get_model_info(model)
    
    dataset = PlantDataset("{}/train_labels.csv".format(opt.label_path), opt.mode, img_resize=opt.img_size)
    train_size = int(len(dataset) * 0.85)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True
    )
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)
    loss = torch.nn.CrossEntropyLoss()
    

    if opt.mode == "train":
        with torch.cuda.device(opt.device):
            epoch_log = []
            train_loss_log = []
            train_acc_log = []
            test_loss_log = []
            test_acc_log = []
            
            id_start_time = time.time()
            log_file =  open("./figure/log_{}.txt".format(int(id_start_time) % 100000), 'w')
            log_file.writelines("start time: {}\n".format(time.ctime(id_start_time)))
            log_file.writelines("===== model config =====\n" +
                            "id: {:.0f}\n".format(int(id_start_time) % 100000) +
                            "input_img_resize: {}\n".format(opt.img_size) + 
                            "epochs: {}\n".format(opt.epoch) +
                            "lr: {}\n".format(opt.lr) +
                            "batch_size: {}\n".format(opt.batch_size))
            for key, value in configs.items():
                log_file.writelines(key + ': ' + str(value) + '\n')
            log_file.writelines("***** remove background *****\n")
            log_file.writelines("========================\n")
            log_file.writelines("eopch, train loss, train acc, test loss, test acc\n")
            
            for epoch in range(opt.epoch):
                print(">>> {}: Epoch {}".format(int(id_start_time) % 100000, epoch+1))
                start_time = time.time()
                
                train_dataset.dataset.set_mode("train")
                train_loss, train_acc = train(model, opt.device, train_loader, optimizer, loss, epoch, id_start_time)
                epoch_log.append(epoch+1)
                train_loss_log.append(train_loss.cpu().detach().numpy())
                train_acc_log.append(train_acc)
                
                test_dataset.dataset.set_mode("test")
                test_loss, test_acc, uncorrect_count = test(model, opt.device, test_loader, loss)
                test_loss_log.append(test_loss.cpu().detach().numpy())
                test_acc_log.append(test_acc)

                plot_uncorrect_count_img(epoch, uncorrect_count, id_start_time)
                plot_loss_img(epoch_log, train_loss_log, test_loss_log, id_start_time)
                plot_acc_img(epoch_log, train_acc_log, test_acc_log, id_start_time)
                log_file.writelines(str(epoch) + ', ' + str(round(train_loss.item(), 4)) + ', ' + str(round(train_acc, 4)) + 
                    ', ' + str(round(test_loss.item(), 4)) + ', ' + str(round(test_acc, 4)) + '\n')
                print("\n>>> Epoch {} end, total times: {}, train loss: {:.6f}, train acc: {:.2f}, test loss: {:.6f}, test acc: {:.2f}".format(
                    epoch+1, round(time.time()-start_time, 4), train_loss, train_acc, test_loss, test_acc))
            log_file.writelines("end time: {}\n".format(time.ctime(time.time())))
            log_file.writelines("===== model structure =====\n")
            log_file.writelines(get_model_info(model, input_size=[opt.img_size, opt.img_size], verbose=2))
    else:
        pass


if __name__ == "__main__":
    main()