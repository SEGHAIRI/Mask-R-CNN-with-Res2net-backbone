import torch
from Model import model
import utils
from engine import train_one_epoch, evaluate
from Data_processing import LoadDataSet, get_train_transform
import argparse

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--TRAIN_PATH', type=str, default='./data_mask/stage1_train/',
                        help='input the data directory')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
print(args.epochs)
print(args)
TRAIN_PATH = args.TRAIN_PATH
dataset = LoadDataSet(TRAIN_PATH, transform=get_train_transform())

torch.manual_seed(args.seed)
indices = torch.randperm(len(dataset)).tolist()
dataset_train = torch.utils.data.Subset(dataset, indices[:int(0.6*len(dataset))])
dataset_test = torch.utils.data.Subset(dataset, indices[int(0.6*len(dataset)):int(0.8*len(dataset))])
dataset_valid = torch.utils.data.Subset(dataset, indices[int(0.8*len(dataset)):])

print('number of train data :', len(dataset_train))
print('number of test data :', len(dataset_test))
print('number of valid data :', len(dataset_valid))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)

data_loader_valid = torch.utils.data.DataLoader(
    dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = torch.device("cuda")
else:
    torch.device('cpu')
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=15,
                                            gamma=0.1)

save_fr = 10
print_freq = 50  # make sure that print_freq is smaller than len(dataset) & len(dataset_test)

for epoch in range(args.epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
    if epoch%save_fr == 0:
        torch.save(model.state_dict(), '../maskrcnn_saved_models/mask_rcnn_model_epoch_{}.pt'.format(str(epoch)))
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
