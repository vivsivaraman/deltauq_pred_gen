#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Delta Encoding based Inferencing

import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
import time

import argparse
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, accuracy_score
import numpy as np

import torch
import torch.nn as nn
from models.modeldefs import resnet18, decoder, correct_incorrect
from data.dataset import *
import pickle


def parse_arguments():
    parser = argparse.ArgumentParser(description='Delta Encoding based Inference')
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str, help='Ckpts directory')
    parser.add_argument('--data_dir', default='./data', type=str, help='Data directory')
    parser.add_argument('--results_dir', default='./results', type=str, help='Results directory')
    parser.add_argument('--dataset_name', default='pacs', type=str, help='Name of the Dataset (Choices: pacs)')
    parser.add_argument('--source_domain', default='photo', type=str, help='Name of the Source Domain (Choices: [photo, art_painting, cartoon, sketch])')
    parser.add_argument('--calibration_domain', default='sketch', type=str, help='Name of the Calibrated Domain (Choices: [photo, art_painting, cartoon, sketch])')
    parser.add_argument('--target_domain', default='sketch', type=str, help='Name of the Target Domain (Choices: [photo, art_painting, cartoon, sketch])')
    parser.add_argument('--num_classes', default=7, type=int, help='Number of classes (Choices: For pacs, 7)')
    parser.add_argument('--model_type', default='resnet18', type=str, help='Classifier model: (Choices:[resnet18, resnet50])')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--correct_incorrect_input_dim', default=256, type=int, help='Input dim to the predictor (Choices: [256, 1])')
    parser.add_argument('--threshold', default=0.6, type=float, help='Threshold for binary detection')
    parser.add_argument('--k', default=20, type=int, help='No. of anchors for inference/validation')
    args = parser.parse_args()

    return args


class InferenceManager():
    """
    Class to manage model inference
    """
    def __init__(self, model=None, decoder=None, predictor=None, source_train_loader=None, source_val_loader=None, target_train_loader=None, target_val_loader=None, device='cpu'):

        self.model = model
        self.decoder = decoder
        self.predictor = predictor
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.target_train_loader = target_train_loader
        self.target_val_loader = target_val_loader
        self.device = device
        self.bce = nn.BCEWithLogitsLoss().to(self.device)  # The NN model should produce logits
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def delta_encoding(self, query, ref):
        return query-ref

    def inference(self):
        """
        Validates the decoder and predictor
        """
        self.model.eval()
        self.decoder.eval()
        self.predictor.eval()
        all_preds_t_list = []

        with torch.no_grad():
            batch_time = 0.
            end = time.time()
            avg_loss = 0.
            true_total_probs_t = []
            pred_total_probs_t = []
            true_total_target_t = []
            pred_total_target_t = []
            total_correct = 0

            # start at a random point of the source dataset; this induces more randomness without obliterating locality
            self.source_train_loader.dataset.offset = np.random.randint(len(self.source_train_loader.dataset))
            source_train_iter = self.source_train_loader.__iter__()

            for i, (x_t, y_t) in enumerate(self.target_train_loader):
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

                all_preds_t = self.get_all_preds(x_t_features, x_s_features, args.k)
                all_preds_t_list.append(all_preds_t)

                mean_pred_logits_t = torch.mean(all_preds_t,axis=0)
                loss = self.bce(mean_pred_logits_t, true_num_correct_t.to(self.device))
                pred_total_probs_t.extend(self.sigmoid(mean_pred_logits_t).data.cpu().numpy())
                ###############################################################

                # Basic logs
                batch_time = time.time() - end
                end = time.time()

                true_total_probs_t.extend(true_probs_t.data.cpu().numpy())
                true_total_target_t.extend(y_t.data.cpu().numpy())
                pred_total_target_t.extend(true_num_correct_t.data.cpu().numpy())

                avg_loss += loss.item()/len(self.target_train_loader)

            true_acc_t = accuracy_score(np.array(true_total_target_t), np.argmax(np.array(true_total_probs_t),1))
            pred_acc_t = ((np.array(pred_total_probs_t) >= args.threshold).sum())/len(np.array(pred_total_probs_t))
            print(len(np.array(pred_total_probs_t)))

            print("Completed Inference\n")
            print("Avg.Target Loss {:.3f}\nTrue_Acc_Target {:.3f}\nPred_Acc_Target {:.3f}\n".format(avg_loss, true_acc_t, pred_acc_t))

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
        np.random.seed(2)
        for i in list(np.random.choice(reference.shape[0], nref)):
            ref = reference[i].to(self.device)
            all_preds.append(self._map_delta_model(ref.expand([n_test,ref.shape[0]]),query.float()))
        all_preds = torch.stack(all_preds).squeeze(2)
        return all_preds


def main():

    # Assigning device (cuda or cpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 1. Loading pre-trained classifier checkpoint
    # Loading the models
    if args.model_type == 'resnet18':
        model = resnet18(ckpt_path=None)
        model.fc = torch.nn.Linear(512, args.num_classes)
        path = os.path.join(args.ckpt_dir, args.model_type, args.dataset_name, args.source_domain, 'ckpt_last.pth.tar')
        model.load_state_dict(torch.load(path, map_location=device)['state_dict'])
        #print(torch.load(path)['epoch'])
        #sys.exit(0)
        print('Pre-trained classifier {} loaded'.format(args.model_type))
        print('Pre-trained clasifier was trained on {}'.format(args.source_domain))
        print(list(model.children()))

    # Step 2. Defining and loading decoder and predictor
    path = os.path.join(args.ckpt_dir, args.model_type, args.dataset_name, args.source_domain, args.calibration_domain,'decoder_predictor_last.pth.tar')
    dec = decoder()
    dec.load_state_dict(torch.load(path, map_location=device)['decoder_state_dict'])
    predictor = correct_incorrect(n_layers=3, input_dim=args.correct_incorrect_input_dim, hidden_dim=512)
    predictor.load_state_dict(torch.load(path, map_location=device)['predictor_state_dict'])
    print('Loaded decoder and predictor')

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

    # Train loader for the given source and target domains
    mode = 'train'
    shuffle=True
    drop_last=True
    tr = transf_val#train
    domain = args.source_domain
    source_train_loader = get_loaders_custom(args.data_dir, args.dataset_name, domain, mode, tr, shuffle, args.batch_size, drop_last)

    mode = 'val'
    shuffle=False
    drop_last=False
    tr = transf_val
    domain = args.source_domain
    source_val_loader = get_loaders_custom(args.data_dir, args.dataset_name, domain, mode, tr, shuffle, args.batch_size, drop_last)

    mode = 'train'
    shuffle=False
    drop_last=True
    tr = transf_val
    domain = args.target_domain
    target_train_loader = get_loaders_custom(args.data_dir, args.dataset_name, domain, mode, tr, shuffle, args.batch_size, drop_last)

    mode = 'val'
    shuffle=False
    drop_last=False
    tr = transf_val
    domain = args.target_domain
    target_val_loader = get_loaders_custom(args.data_dir, args.dataset_name, domain, mode, tr, shuffle, args.batch_size, drop_last)

    print('Loaded source_loader, target_loader')
    manager = InferenceManager(model, dec, predictor, source_train_loader, source_val_loader, target_train_loader, target_val_loader, device)
    manager.inference()


if __name__ == '__main__':
    start = time.time()
    args = parse_arguments()
    main()
    end = time.time()
    time_elapsed = end - start
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
