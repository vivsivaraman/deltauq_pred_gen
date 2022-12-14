#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Train Resnet18 model on Source Domain

import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
import time

import argparse
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, accuracy_score
import numpy as np

import torch, torchvision
import torch.nn as nn
from models.modeldefs import resnet18
from data.dataset import get_loaders


def parse_arguments():
    parser = argparse.ArgumentParser(description='Classifier model')
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str, help='Ckpts directory')
    parser.add_argument('--data_dir', default='./data', type=str, help='Data directory')
    parser.add_argument('--results_dir', default='./results', type=str, help='Results directory')
    parser.add_argument('--dataset_name', default='pacs', type=str, help='Name of the Dataset (Choices: pacs)')
    parser.add_argument('--source_domain', default='photo', type=str, help='Name of the Domain (Choices: [photo, art_painting, cartoon, sketch])')
    parser.add_argument('--num_classes', default=7, type=int, help='Number of classes (Choices: For pacs, 7)')
    parser.add_argument('--model_type', default='resnet18', type=str, help='Classifier model: (Choices:[resnet18, resnet50])')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of epochs to train the classifier')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
    args = parser.parse_args()

    return args


class TrainManager(object):
    """
    Class to manage model training
    :param str model: name of classifier model
    :param DataLoader train_loader: iterate through labeled train data
    :param DataLoader val_loader: iterate through validation data
    :param dict config: dictionary with hyperparameters
    :return: object of TrainManager class
    """
    def __init__(self, model=None, train_loader=None, val_loader=None, test_loader=None, device='cpu'):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=5e-5)

        label_dist = np.unique(self.train_loader.dataset.targets, return_counts=True)[1]
        class_weights = [1 - (x / sum(label_dist)) for x in label_dist]
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        self.softmax = nn.Softmax(dim=1)


    def train(self):
        """
        Trains a pytorch NN classifier model
        :return: None
        """
        current_val_score = -10000
        for epoch in range(1, args.num_epochs+1):
            self.model.train()
            avg_loss = 0.
            total_probs = []
            total_true = []
            total_correct = 0
            batch_time = 0.
            end = time.time()

            for i, (input, target) in enumerate(self.train_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                y_logits, _ = self.model(input)
                loss = self.criterion(y_logits, target)
                probs = self.softmax(y_logits)
                pred = (np.argmax(probs.data.cpu().numpy(),1)).astype(float)
                batch_acc = balanced_accuracy_score(target.data.cpu().numpy(), pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Measure elapsed time
                batch_time = time.time() - end
                end = time.time()

                if i % args.print_freq == 0:
                    print('Epoch: [{}][{}/{}]\t'
                        'Time {:.3f}\t'
                        'Loss {:.4f}\t'
                        'Acc {:.3f}'.format(epoch, i, len(self.train_loader), batch_time, loss.item(), batch_acc))

                total_probs.extend(probs.data.cpu().numpy())
                total_true.extend(target.data.cpu().numpy())
                avg_loss += loss.item()/len(self.train_loader)

            train_acc = balanced_accuracy_score(np.array(total_true), np.argmax(np.array(total_probs),1))
            print("Epoch {}/{}".format(epoch, args.num_epochs))
            print("Avg.Train Loss {:.3f}   Train Acc {:.3f}".format(avg_loss, train_acc))

            # Evaluate on Validation set
            val_score = self.validate(epoch)

            # Save the checkpoint based on best model accuracy
            if val_score > current_val_score:
                current_val_score = val_score
                if(torch.cuda.device_count() > 1):
                    self.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': self.model.module.state_dict(),
                    }, epoch, 'ckpt.pth.tar')
                else:
                    self.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        }, epoch, 'ckpt.pth.tar')

            print('Model Testing')
            self.test()

        # # We also save the last checkpoint and use it for all subsequent experiments
        if(torch.cuda.device_count() > 1):
            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
            }, epoch, 'ckpt_last.pth.tar')
        else:
            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                }, epoch, 'ckpt_last.pth.tar')



    def validate(self, epoch):
        """
        Validates a pytorch nn model
        :return: loss of best model on validation data
        """
        self.model.eval()
        with torch.no_grad():
            batch_time = 0.
            end = time.time()
            total_probs = []
            total_true = []
            total_correct = 0
            avg_loss = 0.0

            for i, (input, target) in enumerate(self.val_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                y_logits, _ = self.model(input)
                loss = self.criterion(y_logits, target)
                probs = self.softmax(y_logits)
                pred = (np.argmax(probs.data.cpu().numpy(),1)).astype(float)
                num_correct = (pred == target.data.cpu().numpy()).sum()
                batch_acc = num_correct/input.size(0)
                total_probs.extend(probs.data.cpu().numpy())
                total_true.extend(target.data.cpu().numpy())

                # Measure elapsed time
                batch_time = time.time() - end
                end = time.time()
                avg_loss += loss.item()/len(self.val_loader)

                if i % args.print_freq == 0:
                    print('Validation: [{}/{}]\t'
                          'Time {:.3f}\t'
                          'Loss {:.4f}\t'
                          'Acc {:.3f}'.format(i, len(self.val_loader), batch_time, loss.item(), batch_acc))


            total_true = np.array(total_true)
            total_probs = np.array(total_probs)
            val_acc = balanced_accuracy_score(total_true, np.argmax(total_probs,1))
            print('Balanced Acc {:.3f}'.format(val_acc))
            print("Confusion Matrix:")
            print(confusion_matrix(total_true, np.argmax(total_probs,1)))

            return val_acc

    def test(self):
        """
        Tests a pytorch nn model
        :return: Metrics on test data
        """
        self.model.eval()
        with torch.no_grad():
            batch_time = 0.
            end = time.time()
            total_probs = []
            total_true = []
            total_correct = 0
            avg_loss = 0.0

            for i, (input, target) in enumerate(self.test_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                y_logits, _ = self.model(input)
                loss = self.criterion(y_logits, target)
                probs = self.softmax(y_logits)
                pred = (np.argmax(probs.data.cpu().numpy(),1)).astype(float)
                num_correct = (pred == target.data.cpu().numpy()).sum()
                batch_acc = num_correct/input.size(0)
                total_probs.extend(probs.data.cpu().numpy())
                total_true.extend(target.data.cpu().numpy())
                total_correct += num_correct

                # Measure elapsed time
                batch_time = time.time() - end
                end = time.time()
                avg_loss += loss.item()/len(self.test_loader)

                if i % args.print_freq == 0:
                    print('Test: [{}/{}]\t'
                          'Time {:.3f}\t'
                          'Loss {:.4f}\t'
                          'Acc {:.3f}'.format(i, len(self.test_loader), batch_time, loss.item(), batch_acc))

            total_true = np.array(total_true)
            total_probs = np.array(total_probs)
            val_acc = balanced_accuracy_score(total_true, np.argmax(total_probs,1))
            print('Balanced Acc {:.3f}'.format(val_acc))
            print("Confusion Matrix:")
            print(confusion_matrix(total_true, np.argmax(total_probs,1)))


    def save_checkpoint(self, state, epoch, name):
        """Saves checkpoint to disk"""
        print('Saving model at Epoch {}'.format(epoch))
        directory = os.path.join(args.ckpt_dir, args.model_type, args.dataset_name, args.source_domain)
        filename = directory + '/' + name
        torch.save(state, filename)

def main():

    # Assigning device (cuda or cpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Creating necessary filepaths and folders
    if not os.path.exists(os.path.join(args.ckpt_dir, args.model_type, args.dataset_name, args.source_domain)):
        os.makedirs(os.path.join(args.ckpt_dir, args.model_type, args.dataset_name, args.source_domain))
    if not os.path.exists(os.path.join(args.results_dir, args.model_type,  args.dataset_name, args.source_domain)):
        os.makedirs(os.path.join(args.results_dir, args.model_type, args.dataset_name, args.source_domain))

    # Train and Val loaders for the given source domain
    train_loader, val_loader, test_loader = get_loaders(args.data_dir, args.dataset_name, args.source_domain, args.batch_size)
    print('Loaded train, val and test loaders')

    # Loading the model
    if args.model_type == 'resnet18':
        #resnet18-f37072fd.pth is the pre-trained, resnet18 imagenet checkpoint downloaded from Pytorch website
        path = os.path.join(args.ckpt_dir, 'resnet18-f37072fd.pth')
        model = resnet18(path)
        model.fc = torch.nn.Linear(512, args.num_classes)
        print('Model chosen: {}'.format(args.model_type))
        print(list(model.children()))


    # Multi-gpu training if available
    if(torch.cuda.device_count() > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    manager = TrainManager(model, train_loader, val_loader, test_loader, device)
    manager.train()


if __name__ == '__main__':
    start = time.time()
    args = parse_arguments()
    main()
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
