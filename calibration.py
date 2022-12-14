#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Delta Encoding based Calibration

import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
import time

import argparse
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, balanced_accuracy_score, accuracy_score
import numpy as np

import torch
import torch.nn as nn
from models.modeldefs import resnet18, decoder, correct_incorrect
from data.dataset import *


def parse_arguments():
    parser = argparse.ArgumentParser(description='Delta Encoding based Calibration')
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str, help='Ckpts directory')
    parser.add_argument('--data_dir', default='./data', type=str, help='Data directory')
    parser.add_argument('--results_dir', default='./results', type=str, help='Results directory')
    parser.add_argument('--dataset_name', default='pacs', type=str, help='Name of the Dataset (Choices: pacs)')
    parser.add_argument('--source_domain', default='photo', type=str, help='Name of the Source Domain (Choices: [photo, art_painting, cartoon, sketch])')
    parser.add_argument('--calibration_domain', default='sketch', type=str, help='Name of the calibration Domain (Choices: [photo, art_painting, cartoon, sketch])')
    parser.add_argument('--num_classes', default=7, type=int, help='Number of classes (Choices: For pacs, 7)')
    parser.add_argument('--model_type', default='resnet18', type=str, help='Classifier model: (Choices:[resnet18])')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of epochs to train the decoder/correct incorrect classifier/error predictor')
    parser.add_argument('--learning_rate', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--correct_incorrect_input_dim', default=256, type=int, help='Input dim to the predictor (Choices: [256, 1])')
    parser.add_argument('--threshold', default=0.40, type=float, help='Threshold for binary detection')
    parser.add_argument('--k', default=20, type=int, help='No. of anchors for inference/validation')
    parser.add_argument('--tau1', default=0.25, type=float, help='Threshold for binary detection')
    parser.add_argument('--tau2', default=0.65, type=float, help='Threshold for binary detection')
    args = parser.parse_args()
    return args

class TrainManager():
    """
    Class to manage model training
    """
    def __init__(self, model=None, decoder=None, predictor=None, source_train_loader=None, source_val_loader=None, calibration_train_loader=None, calibration_val_loader=None, device='cpu'):

        self.model = model
        self.decoder = decoder
        self.predictor = predictor
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.calibration_train_loader = calibration_train_loader
        self.calibration_val_loader = calibration_val_loader
        self.device = device
        params = list(self.decoder.parameters())+list(self.predictor.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=5e-5)
        self.bce = nn.BCEWithLogitsLoss().to(self.device)  # The NN model should produce logits
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def delta_encoding(self, query, ref):
        return query-ref

    def train(self):
        """
        Trains a pytorch decoder + correct/incorrect model
        """
        current_val_score = -10000
        for epoch in range(1, args.num_epochs+1):
            print('###########################################################')
            print('Training')
            self.model.eval()
            self.decoder.train()
            self.predictor.train()
            avg_loss = 0.

            true_total_probs_t = []
            pred_total_probs_t = []
            true_total_calibration_t = []
            pred_total_calibration_t = []

            total_correct = 0
            batch_time = 0.
            end = time.time()

            # Start at a random point of the source dataset; this induces more randomness without obliterating locality
            self.source_train_loader.dataset.offset = np.random.randint(len(self.source_train_loader.dataset))
            source_train_iter = self.source_train_loader.__iter__()

            for i, (x_t, y_t) in enumerate(self.calibration_train_loader):
                ###############################################################
                try:
                    x_s, y_s = next(source_train_iter)
                except StopIteration:
                    del source_train_iter
                    source_train_iter = self.source_train_loader.__iter__()
                    x_s, y_s = next(source_train_iter)

                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)
                y_logits, features = self.model(torch.cat([x_s, x_t], 0))

                y_s_logits = y_logits[:len(x_s)]
                y_t_logits = y_logits[len(x_s):]
                x_s_features = features[:len(x_s)]
                x_t_features = features[len(x_s):]

                # Delta encoding --> Decoder --> Predictor
                delta = self.delta_encoding(x_t_features, x_s_features)
                decoder_inp = torch.cat([x_s_features, delta],axis=1)
                decoder_out = self.decoder(decoder_inp)
                predictor_inp = torch.abs(x_t_features-decoder_out)
                predictor_out = self.predictor(predictor_inp)

                # Preparation of pseudo labels to train the decoder and predictor
                true_probs_t = self.softmax(y_t_logits)
                true_probs_t_slices = true_probs_t.gather(1,y_t.unsqueeze(1))
                tmp = []
                for k in range(len(true_probs_t_slices)):
                    if true_probs_t_slices[k] >= args.tau2:
                        tmp.append(1.0)
                    elif true_probs_t_slices[k] <= args.tau1:
                        tmp.append(0.0)
                    else:
                        tmp.append(0.5)
                tmp = np.array(tmp)
                true_num_correct_t = torch.FloatTensor(tmp).to(self.device)
                del tmp

                # Compute BCE
                loss = self.bce(predictor_out.squeeze(1), true_num_correct_t.to(self.device))
                pred_total_probs_t.extend(self.sigmoid(predictor_out).data.cpu().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ###############################################################

                # Basic logs
                batch_time = time.time() - end
                end = time.time()
                true_total_probs_t.extend(true_probs_t.data.cpu().numpy())
                true_total_calibration_t.extend(y_t.data.cpu().numpy())
                pred_total_calibration_t.extend(true_num_correct_t.data.cpu().numpy())
                avg_loss += loss.item()/len(self.calibration_train_loader)

            true_train_acc_t = accuracy_score(np.array(true_total_calibration_t), np.argmax(np.array(true_total_probs_t),1))
            pred_train_acc_t = ((np.array(pred_total_probs_t) >= args.threshold).sum())/len(np.array(pred_total_probs_t))

            print("Epoch {}/{}".format(epoch, args.num_epochs))
            print("Avg.Train Loss {:.3f} | True_Tr_Acc_Cal {:.3f} | Pred_Tr_Acc_Cal {:.3f}".format(avg_loss, true_train_acc_t, pred_train_acc_t))
            print('Validation')

            # Evaluation on validation set
            val_score = self.validate(epoch)

            # Save checkpoint based on validation accuracy
            if val_score > current_val_score:
                current_val_score = val_score
                print('Saving models at epoch {}'.format(epoch))
                if(torch.cuda.device_count() > 1):
                    self.save_checkpoint({
                        'epoch': epoch,
                        'decoder_state_dict': self.decoder.module.state_dict(),
                        'predictor_state_dict': self.predictor.module.state_dict(),
                    }, epoch, 'decoder_predictor.pth.tar')
                else:
                    self.save_checkpoint({
                        'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'predictor_state_dict': self.predictor.state_dict(),
                    }, epoch, 'decoder_predictor.pth.tar')

        # save model for last epoch
        if(torch.cuda.device_count() > 1):
            self.save_checkpoint({
                'epoch': epoch,
                'decoder_state_dict': self.decoder.module.state_dict(),
                'predictor_state_dict': self.predictor.module.state_dict(),
            }, epoch, 'decoder_predictor_last.pth.tar')
        else:
            self.save_checkpoint({
                'epoch': epoch,
                'decoder_state_dict': self.decoder.state_dict(),
                'predictor_state_dict': self.predictor.state_dict(),
            }, epoch, 'decoder_predictor_last.pth.tar')


    def validate(self, epoch):
        """
        Validates the decoder and correct vs incorrect predictor
        """
        self.model.eval()
        self.decoder.eval()
        self.predictor.eval()
        with torch.no_grad():
            batch_time = 0.
            end = time.time()
            avg_loss = 0.
            true_total_probs_t = []
            pred_total_probs_t = []
            true_total_calibration_t = []
            pred_total_calibration_t = []
            total_correct = 0

            # start at a random point of the source dataset; this induces more randomness without obliterating locality
            self.source_train_loader.dataset.offset = np.random.randint(len(self.source_train_loader.dataset))
            source_train_iter = self.source_train_loader.__iter__()

            for i, (x_t, y_t) in enumerate(self.calibration_val_loader):
                ###############################################################
                try:
                    x_s, y_s = next(source_train_iter)
                except StopIteration:
                    del source_train_iter
                    source_train_iter = self.source_train_loader.__iter__()
                    x_s, y_s = next(source_train_iter)

                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)
                y_logits, features = self.model(torch.cat([x_s, x_t], 0))

                y_s_logits = y_logits[:len(x_s)]
                y_t_logits = y_logits[len(x_s):]
                x_s_features = features[:len(x_s)]
                x_t_features = features[len(x_s):]

                true_probs_t = self.softmax(y_t_logits)
                true_pred_t = (np.argmax(true_probs_t.data.cpu().numpy(),1)).astype(float)
                true_num_correct_t = (true_pred_t == y_t.data.cpu().numpy()).astype(float)
                true_num_correct_t = torch.FloatTensor(true_num_correct_t).to(self.device)

                # Delta UQ based inferencing (marginalization)
                all_preds_t = self.get_all_preds(x_t_features, x_s_features, args.k)
                mean_pred_logits_t = torch.mean(all_preds_t,axis=0)
                #var_pred_logits_t = torch.var(all_preds_t,axis=0)
                loss = self.bce(mean_pred_logits_t, true_num_correct_t.to(self.device))
                pred_total_probs_t.extend(self.sigmoid(mean_pred_logits_t).data.cpu().numpy())
                ###############################################################

                # Basic logs
                batch_time = time.time() - end
                end = time.time()

                true_total_probs_t.extend(true_probs_t.data.cpu().numpy())
                true_total_calibration_t.extend(y_t.data.cpu().numpy())
                pred_total_calibration_t.extend(true_num_correct_t.data.cpu().numpy())

                avg_loss += loss.item()/len(self.calibration_val_loader)

            true_val_acc_t = accuracy_score(np.array(true_total_calibration_t), np.argmax(np.array(true_total_probs_t),1))
            pred_val_acc_t = ((np.array(pred_total_probs_t) >= args.threshold).sum())/len(np.array(pred_total_probs_t))

            print("Epoch {}/{}".format(epoch, args.num_epochs))
            print("Avg.Val Loss {:.3f}| True_Val_Acc_Cal {:.3f} |Pred_Val_Acc_Cal {:.3f}".format(avg_loss, true_val_acc_t, pred_val_acc_t))

            return pred_val_acc_t



    def _map_delta_model(self, ref, query):
        delta = self.delta_encoding(query, ref)
        decoder_inp = torch.cat([ref,delta],1)
        decoder_out = self.decoder(decoder_inp)
        predictor_inp = torch.abs(query-decoder_out)
        predictor_out = self.predictor(predictor_inp)
        return predictor_out

    def get_all_preds(self, query, reference, k=20):
        nref=np.minimum(k, reference.shape[0])
        all_preds = []
        n_test = query.shape[0]
        for i in list(np.random.choice(reference.shape[0], nref)):
            ref = reference[i].to(self.device)
            all_preds.append(self._map_delta_model(ref.expand([n_test,ref.shape[0]]),query.float()))

        all_preds = torch.stack(all_preds).squeeze(2)
        return all_preds


    def save_checkpoint(self, state, epoch, name):
        """Saves checkpoint to disk"""
        print('Saving model at Epoch {}'.format(epoch))
        directory = os.path.join(args.ckpt_dir, args.model_type, args.dataset_name, args.source_domain, args.calibration_domain)
        filename = directory + '/' + name
        torch.save(state, filename)

def main():

    # Assigning device (cuda or cpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Creating necessary filepaths and folders
    if not os.path.exists(os.path.join(args.ckpt_dir, args.model_type, args.dataset_name, args.source_domain, args.calibration_domain)):
        os.makedirs(os.path.join(args.ckpt_dir, args.model_type, args.dataset_name, args.source_domain, args.calibration_domain))
    if not os.path.exists(os.path.join(args.results_dir, args.model_type,  args.dataset_name, args.source_domain, args.calibration_domain)):
        os.makedirs(os.path.join(args.results_dir, args.model_type, args.dataset_name, args.source_domain, args.calibration_domain))

    # Step 1. Loading pre-trained classifier checkpoint
    # Loading the models
    if args.model_type == 'resnet18':
        model = resnet18(ckpt_path=None)
        model.fc = torch.nn.Linear(512, args.num_classes)
        path = os.path.join(args.ckpt_dir, args.model_type, args.dataset_name, args.source_domain, 'ckpt_last.pth.tar')
        model.load_state_dict(torch.load(path)['state_dict'])
        print('Pre-trained classifier {} loaded'.format(args.model_type))
        print('Pre-trained clasifier was trained on {}'.format(args.source_domain))
        print(list(model.children()))

    # Step 2. Defining decoder and predictor (correct vs incorrect estimator)
    dec = decoder()
    predictor = correct_incorrect(n_layers=3, input_dim=args.correct_incorrect_input_dim, hidden_dim=512)

    # Multi-gpu training if available
    if(torch.cuda.device_count() > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        dec = nn.DataParallel(dec)
        predictor = nn.DataParallel(predictor)

    model = model.to(device)
    dec = dec.to(device)
    predictor = predictor.to(device)

    # Get the number of model parameters
    print('Number of Classifier Model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('Number of Decoder Model parameters: {}'.format(
        sum([p.data.nelement() for p in dec.parameters()])))
    print('Number of Predictor Model parameters: {}'.format(
        sum([p.data.nelement() for p in predictor.parameters()])))


    # Train & Val loader for the given source and calibration domains
    mode = 'train'
    shuffle=True
    drop_last=True
    tr = transf_train
    domain = args.source_domain
    source_train_loader = get_loaders_custom(args.data_dir, args.dataset_name, domain, mode, tr, shuffle, args.batch_size, drop_last)

    mode = 'val'
    shuffle=False
    drop_last=False
    tr = transf_val
    domain = args.source_domain
    source_val_loader = get_loaders_custom(args.data_dir, args.dataset_name, domain, mode, tr, shuffle, args.batch_size, drop_last)

    mode = 'train'
    shuffle=True
    drop_last=True
    tr = transf_train
    domain = args.calibration_domain
    calibration_train_loader = get_loaders_custom(args.data_dir, args.dataset_name, domain, mode, tr, shuffle, args.batch_size, drop_last)

    mode = 'val'
    shuffle=False
    drop_last=False
    tr = transf_val
    domain = args.calibration_domain
    calibration_val_loader = get_loaders_custom(args.data_dir, args.dataset_name, domain, mode, tr, shuffle, args.batch_size, drop_last)
    print('Loaded source_loader, calibration_loader')

    manager = TrainManager(model, dec, predictor, source_train_loader, source_val_loader, calibration_train_loader, calibration_val_loader, device)
    manager.train()


if __name__ == '__main__':
    start = time.time()
    args = parse_arguments()
    main()
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
