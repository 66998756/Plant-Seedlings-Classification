import os
from os import listdir
import csv


def to_csv(mode):
    path = os.path.join('dataset', mode)
    labels = listdir(path)
    
    data_dict = {}
    for label in labels:
        datas = listdir(os.path.join(path, label))
        for data in datas:
            if label in data_dict.keys():
                data_dict[label].append(data)
            else:
                data_dict[label] = []
    
    print(data_dict.keys())
    # with open('{}_labels.csv'.format(mode), 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["file", "species"])
    #     for label, imgs in data_dict.items():
    #         for img in imgs:
    #             writer.writerow([img, label])


if __name__ == "__main__":
    to_csv("train")
