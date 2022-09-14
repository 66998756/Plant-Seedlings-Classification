import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1, help='gpu device number')
    parser.add_argument('--data_path', default="", help='path of dataset')
    parser.add_argument('--label_path', default=".", help='path of dataset')
    parser.add_argument('--mode', default='train', choices=['train', 'evaluate', 'vis'], help='train or evaluate ')
    parser.add_argument("--img_size", type=int, default=192, help='model input size')
    parser.add_argument('--epoch', type=int, default=111, help='number of epoch to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning_rate')
    parser.add_argument("--batch_size", type=int, default=2, help='the batch for id')

    opt = parser.parse_args()
    return opt