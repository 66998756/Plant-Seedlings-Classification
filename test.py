from tqdm import tqdm

import torch


def test(model, device, test_loader, loss):
    model.eval()
    test_loss = 0
    test_acc = 0
    uncorrect_count = [0 for i in range(12)]
    with torch.no_grad():
        item = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output)
            l = loss(output, target)
            global test_loss += l
            test_acc += (output.argmax(dim=1) == target.argmax(dim=1)).float().sum().item()
            
            for idx, target_ in enumerate(torch.argmax(target, dim=1)):
                # print(target.tolist())
                if torch.argmax(output[idx]).item() != target_.item():
                    uncorrect_count[target_.item()] += 1
            
            print('\rTesting: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(data), len(test_loader.dataset),
            100. * batch_idx / len(test_loader), l), end='')
            
    return test_loss / len(test_loader), test_acc / len(test_loader.dataset), uncorrect_count