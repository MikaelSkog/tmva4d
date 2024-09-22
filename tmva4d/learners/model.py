"""Class to train a PyTorch model"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tmva4d.loaders.dataloaders import Dataset4dDataset
from tmva4d.learners.tester import Tester
from tmva4d.utils.functions import normalize, define_loss, get_transformations
from tmva4d.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer


class Model(nn.Module):
    """Class to train a model

    PARAMETERS
    ----------
    net: PyTorch Model
        Network to train
    data: dict
        Parameters and configurations for training
    """

    def __init__(self, net, data):
        super().__init__()
        self.net = net
        self.cfg = data['cfg']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.model_name = self.cfg['model']
        self.process_signal = self.cfg['process_signal']
        self.annot_type = self.cfg['annot_type']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.loss_step = self.cfg['loss_step']
        self.val_step = self.cfg['val_step']
        self.viz_step = self.cfg['viz_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.writer = SummaryWriter(self.paths['writer'])
        self.visualizer = TensorboardMultiLossVisualizer(self.writer)
        self.tester = Tester(self.cfg, self.visualizer)
        self.results = dict()

    def train(self, add_temp=False):
        """
        Method to train a network

        PARAMETERS
        ----------
        add_temp: boolean
            Add a temporal dimension during training?
            Considering the input as a sequence.
            Default: False
        """
        self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        self._set_seeds()
        self.net.apply(self._init_weights)
        ea_criterion = define_loss('elevation_azimuth', self.custom_loss, self.device)
        nb_losses = len(ea_criterion)
        running_losses = list()
        ea_running_losses = list()
        ea_running_global_losses = [list(), list()]
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        iteration = 0
        best_val_dice = 0
        self.net.to(self.device)

        for epoch in range(self.nb_epochs):
            if epoch % self.lr_step == 0 and epoch != 0:
                scheduler.step()
            for _, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                path_to_frames = os.path.join(self.paths['dataset4d'], seq_name[0])
                frame_dataloader = DataLoader(Dataset4dDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp),
                                              shuffle=self.is_shuffled,
                                              batch_size=self.batch_size,
                                              num_workers=4)
                for _, frame in enumerate(frame_dataloader):
                    ea_data = frame['ea_matrix'].to(self.device).float()
                    da_data = frame['da_matrix'].to(self.device).float()
                    ed_data = frame['ed_matrix'].to(self.device).float()
                    er_data = frame['er_matrix'].to(self.device).float()
                    ra_data = frame['ra_matrix'].to(self.device).float()
                    ea_mask = frame['ea_mask'].to(self.device).float()
                    ea_data = normalize(ea_data, 'elevation_azimuth', norm_type=self.norm_type)
                    da_data = normalize(da_data, 'doppler_azimuth', norm_type=self.norm_type)
                    ed_data = normalize(ed_data, 'elevation_doppler', norm_type=self.norm_type)
                    er_data = normalize(er_data, 'elevation_range', norm_type=self.norm_type)
                    ra_data = normalize(ra_data, 'range_azimuth', norm_type=self.norm_type)
                    optimizer.zero_grad()

                    ea_outputs = self.net(ea_data, da_data, ed_data, er_data, ra_data)
                    ea_outputs = ea_outputs.to(self.device)
                    
                    ea_losses = [c(ea_outputs, torch.argmax(ea_mask, axis=1))
                                    for c in ea_criterion]
                    ea_loss = torch.mean(torch.stack(ea_losses))
                    loss = torch.mean(ea_loss)

                    loss.backward()
                    optimizer.step()
                    running_losses.append(loss.data.cpu().numpy()[()])
                    ea_running_losses.append(ea_loss.data.cpu().numpy()[()])
                    ea_running_global_losses[0].append(ea_losses[0].data.cpu().numpy()[()])
                    # Case with both sDice and CE losses
                    if nb_losses > 1:
                        ea_running_global_losses[1].append(ea_losses[1].data.cpu().numpy()[()])

                    if iteration % self.loss_step == 0:
                        train_loss = np.mean(running_losses)
                        ea_train_loss = np.mean(ea_running_losses)
                        ea_train_losses = [np.mean(sub_loss) for sub_loss in ea_running_global_losses]
                        print('[Epoch {}/{}, iter {}]: '
                              'train loss {}'.format(epoch+1,
                                                     self.nb_epochs,
                                                     iteration,
                                                     train_loss))
                        
                        self.visualizer.update_multi_train_loss(train_loss, ea_train_loss,
                                                                ea_train_losses, iteration)
                        running_losses = list()
                        ea_running_losses = list()
                        self.visualizer.update_learning_rate(scheduler.get_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration,
                                                              add_temp=add_temp)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                        self.visualizer.update_multi_val_metrics(val_metrics, iteration)
                        print('[Epoch {}/{}] Validation losses: '
                              'EA={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    val_metrics['elevation_azimuth']['loss']))
                        print('[Epoch {}/{}] Validation Dice: '
                              'EA={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    val_metrics['elevation_azimuth']['dice']))

                        if val_metrics['global_dice'] > best_val_dice and iteration > 0:
                            best_val_dice = val_metrics['global_dice']
                            test_metrics = self.tester.predict(self.net, test_loader,
                                                               add_temp=add_temp)
                            print('[Epoch {}/{}] Test losses: '
                                  'EA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['elevation_azimuth']['loss']))
                            print('[Epoch {}/{}] Test Dice: '
                                  'EA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['elevation_azimuth']['dice']))

                            self.results['ea_train_loss'] = ea_train_loss.item()
                            self.results['train_loss'] = train_loss.item()
                            self.results['val_metrics'] = val_metrics
                            self.results['test_metrics'] = test_metrics
                            self._save_results()
                        self.net.train()  # Train mode after evaluation process
                    iteration += 1
        self.writer.close()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0., 1.)
                nn.init.constant_(m.bias, 0.)

    def _save_results(self):
        results_path = self.paths['results'] / 'results.json'
        model_path = self.paths['results'] / 'model.pt'
        with open(results_path, "w") as fp:
            json.dump(self.results, fp)
        torch.save(self.net.state_dict(), model_path)

    def _set_seeds(self):
        torch.cuda.manual_seed_all(self.torch_seed)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
