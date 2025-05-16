import argparse
import csv
import os
import sys
import shutil
cdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cdir, 'MolCLR'))
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset, DataLoader
from rdkit import Chem

import finetune
from dataset.dataset_test import MolTestDatasetWrapper, MolTestDataset

apex_support = False

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        # shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))

def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row['smiles']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)
            if mol != None and label != '':
                smiles_data.append(smiles)
                if task == 'classification':
                    labels.append(int(label))
                elif task == 'regression':
                    labels.append(float(label))
                else:
                    ValueError('task must be either regression or classification')
    print('Fixed', len(smiles_data))
    return smiles_data, labels

class FineTuneReturnsPredictedValues(finetune.FineTune):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        log_dir = os.path.join(cdir, 'MolCLR', 'ckpt', 'finetune', os.path.basename(self.writer.log_dir))
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(cdir, 'MolCLR', 'ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model
    
    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d, __ in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = finetune.Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = finetune.amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with finetune.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        y_pred_generator = lambda loader: {
            'indices': loader.batch_sampler.sampler.indices,
            'predictions': self._test_with_predicted_values(model, loader)
        }
        
        return(y_pred_generator(train_loader), y_pred_generator(valid_loader), y_pred_generator(test_loader))

    def _test_with_predicted_values(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        return np.array(predictions)
    

class MolTestDatasetWithFixingSmiles(MolTestDataset):
    def __init__(self, data_path, target, task):
        super().__init__(data_path, target, task)
        self.smiles_data, self.labels = read_smiles(data_path, target, task)


class MolTestDatasetWrapperWithCustomSplit(MolTestDatasetWrapper):
    def __init__(self, 
            batch_size, num_workers, valid_size, test_size, 
            data_path, target, task, splitting):
        super().__init__( 
            batch_size, num_workers, None, None, 
            data_path, target, task, splitting
            )
    
    def get_data_loaders(self):
        train_dataset = MolTestDatasetWithFixingSmiles(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        indices = pd.read_csv(self.data_path)['split'].to_list()
        valid_idx = np.array([i for i, x in enumerate(indices) if x == 'val'])
        test_idx = np.array([i for i, x in enumerate(indices) if x == 'test'])
        train_idx = np.array([i for i, x in enumerate(indices) if x == 'train'])

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader

def fine_tuning(config):
    dataset = MolTestDatasetWrapperWithCustomSplit(config['batch_size'], **config['dataset'])

    fine_tune = FineTuneReturnsPredictedValues(dataset, config)
    return fine_tune.train()
    
def main(args):
    config_path = args.config_path
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    config['dataset']['task'] = 'regression'
    config['dataset']['data_path'] = args.data_path
    target_list = ['obj']

    print(config)

    results_list = []
    for target in target_list:
        config['dataset']['target'] = target
        train, val, test = fine_tuning(config)
    
    for dataset in [train, val, test]:
        indices = dataset['indices']
        predictions = dataset['predictions']
        df_part = pd.DataFrame({'indice': indices, 'prediction': predictions.ravel()})
        results_list.append(df_part)

    df = pd.concat(results_list).sort_values(by='indice').reset_index(drop=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MolCLR Fine-tuning')
    parser.add_argument('--config_path', type=str, default=os.path.join(cdir,'config_finetune.yaml'), help='Path to the config file')
    parser.add_argument('--data_path', type=str, default=os.path.join(cdir,'test.csv'), help='Path to the data file')
    args = parser.parse_args()

    preds = main(args)
    preds.to_csv('predictions.csv', index=False)
    # meta  = pd.read_csv(args.data_path)
    # concat = pd.concat([meta, preds], axis=1, ignore_index=True)
    # concat.to_csv('predictions.csv', index=False)
