import torch

from model import ResNeXt

def train(model, device, train_loader, optimizer, loss, epoch, id_start_time):
    model.train()
    
    train_loss, train_acc = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(data)
        optimizer.zero_grad()
        output = model(data)
        l = loss(output, target)
        l.backward()
        optimizer.step()
        
        print('\rTraining: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), l), end='')

        train_acc += (output.argmax(dim=1) == target.argmax(dim=1)).float().sum().item()
        train_loss += l
    
    if epoch % 10 == 0 and epoch >= 30:
        torch.save(model, "./checkpoint/model_{}_{}.pt".format(int(id_start_time) % 100000, epoch))
    return train_loss / len(train_loader), train_acc / len(train_loader.dataset)
        

def test(model, device, test_loader, loss):
    model.eval()
    valid_loss = 0
    test_acc = 0
    test_loss = 0
    uncorrect_count = [0 for i in range(12)]
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output)
            l = loss(output, target)
            test_loss += l
            test_acc += (output.argmax(dim=1) == target.argmax(dim=1)).float().sum().item()
            
            for idx, target_ in enumerate(torch.argmax(target, dim=1)):
                # print(target.tolist())
                if torch.argmax(output[idx]).item() != target_.item():
                    uncorrect_count[target_.item()] += 1
            
            print('\rTesting: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(data), len(test_loader.dataset),
            100. * batch_idx / len(test_loader), l), end='')
            
    return test_loss / len(test_loader), test_acc / len(test_loader.dataset), uncorrect_count


if __name__ == "__main__":
    model = ResNeXt(num_blocks=[3, 3, 3], cardinality=4, bottleneck_width=4)
    x = torch.rand(1, 3, 32, 32)
    y = torch.rand(1, 12)

