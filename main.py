import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import argparse
import numpy as np
import csv

from dataLoader import read_UTKinect, UTKinectDataset, CLASS_SIZE
from stlstm import StackedSTLSTM, STLSTMLayer, STLSTMCell_wTrustGate, STLSTMCell

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--num_sub_seq', default=10, type=int, help='Number of sub-sequences')
parser.add_argument('--dataset_root', default='./data/UTKinect', type=str, help='dataset root')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--input_size', default=3, type=int, help='input size')
parser.add_argument('--num_layers', default=2, type=int, help='number of layers')
parser.add_argument('--hidden_size', default=32, type=int, help='hidden state size')
parser.add_argument('--with_trust_gate', default='Y', choices=['Y', 'N'], help='type of STLSTM cell')
parser.add_argument('--tree_traversal', default='Y', choices=['Y', 'N'], help='with tree traversal or not')
parser.add_argument('--learning_rate', default=1e-2, type=float, help='learning rate')
parser.add_argument('--end_factor', default=1e-2, type=float, help='end_factor of linear scheduler')
parser.add_argument('--total_iters', default=100, type=int, help='total_iters of linear scheduler')

args = parser.parse_args()

# read dataset
videos, labels = read_UTKinect(args.dataset_root, args.num_sub_seq)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# joints order
if args.tree_traversal == "Y":
    joints_order = [2,3,4,3,5,6,7,8,7,6,5,3,9,10,11,12,11,10,9,3,2,
                    1,13,14,15,16,15,14,13,1,17,18,19,20,19,18,17,1,2]
else:
    joints_order = [i for i in range(1,21)]

# loss function
nllloss = nn.NLLLoss()
def loss_fn(pred, target):
    target = target.unsqueeze(-1).unsqueeze(-1)
    target = target.expand(-1, pred.shape[2], pred.shape[3])
    return nllloss(pred, target)


overall_val_acc = 0
result_file = open('result.csv', 'w+', newline='')
writer = csv.writer(result_file)
writer.writerow(['validation index', 'epoch', 'training loss', 'training accuracy', 
                 'validation loss', 'validation accuracy'])
for v in range(len(videos)):

    # Prepare dataset by leave-one-out-cross-validation (LOOCV)
    # Each video will be used for validation one by one
    print('====================================================')
    print(f'### Cross validation ({v+1} / {len(videos)}) ###')
    val_index = [v]
    train_index = [i for i in range(len(labels)) if i not in val_index]
    train_dataset = UTKinectDataset(videos, labels, train_index, joints_order)
    val_dataset = UTKinectDataset(videos, labels, val_index, joints_order)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    
    # prepare a new model
    model = StackedSTLSTM(args.num_layers, STLSTMLayer, STLSTMCell_wTrustGate if args.with_trust_gate == 'Y' 
                          else STLSTMCell, args.input_size, args.hidden_size, CLASS_SIZE, device).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, 
                    end_factor=args.end_factor, total_iters=args.total_iters)

    # gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)

    best_train_loss = np.inf
    for epoch in range(args.num_epochs):
        print(f'Epoch #{epoch + 1}')

        ### TRAIN
        model.train()
        train_loss = 0.
        train_acc = 0
        train_n = 0
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, target) in progress_bar:

            model.zero_grad()
            inputs = inputs.to(device)
            target = target.to(device)
            log_prob, pred = model(inputs)

            # loss calculation and backproprogate
            loss = loss_fn(log_prob, target)
            train_n += len(target)
            train_loss += loss.data.item() * len(target)
            loss.backward()
            optimizer.step()

            # accuracy
            train_acc += (pred == target).sum().item()

        train_loss /= train_n
        train_acc /= train_n
        print(f'Training loss: {train_loss:.2f}; Accuracy: {train_acc:.2%}')
        
        # save the best model so far
        if best_train_loss > train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), 'model_best.pt')

        ### VALIDATION
        with torch.no_grad():
            val_loss = 0.
            val_acc = 0
            val_n = 0
            model.eval()
            for (inputs, target) in val_loader:
                inputs = inputs.to(device)
                target = target.to(device)
                log_prob, pred = model(inputs)

                # loss calculation
                loss = loss_fn(log_prob, target)
                val_n += len(target)
                val_loss += loss.data.item() * len(target)

                # accuracy
                val_acc += (pred == target).sum().item()

            val_loss /= val_n
            val_acc /= val_n
            print(f'Validation loss: {val_loss:.2f}; Accuracy: {val_acc:.2%}')

        # write result to csv
        writer.writerow([v+1, epoch+1, train_loss, train_acc, val_loss, val_acc])

    print('====================================================')
    print('### Validation using the best model ###')
    # Recall the best model
    state_dict = torch.load('model_best.pt')
    model = StackedSTLSTM(args.num_layers, STLSTMLayer, STLSTMCell_wTrustGate if args.with_trust_gate == 'Y' 
                          else STLSTMCell, args.input_size, args.hidden_size, CLASS_SIZE, device).to(device)
    model.load_state_dict(state_dict)

    # do validation using the best model
    with torch.no_grad():
        model.eval()
        val_acc = 0
        val_n = 0
        for (inputs, target) in val_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            _, pred = model(inputs)
            # accuracy
            val_n += len(target)
            val_acc += (pred == target).sum().item()
        val_acc /= val_n
        print(f'Validation accuracy: {val_acc:.2%}')

    overall_val_acc += val_acc
    
# Report final LOOCV accuracy
print('====================================================')
print(f'FINAL VALIDATION ACCURACY: {overall_val_acc / len(videos):.2%}')