"""Class to test a model"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from tmva4d.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, get_qualitatives
from tmva4d.utils.paths import Paths
from tmva4d.utils.metrics import Evaluator
from tmva4d.loaders.dataloaders import Dataset4dDataset


class Tester:
    """
    Class to test a model

    PARAMETERS
    ----------
    cfg: dict
        Configuration parameters used for train/test
    visualizer: object or None
        Add a visulization during testing
        Default: None
    """

    def __init__(self, cfg, visualizer=None):
        self.cfg = cfg
        self.visualizer = visualizer
        self.model = self.cfg['model']
        self.nb_classes = self.cfg['nb_classes']
        self.annot_type = self.cfg['annot_type']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.n_frames = self.cfg['nb_input_channels']
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.paths = Paths().get()
        self.test_results = dict()

    def predict(self, net, seq_loader, iteration=None, get_quali=False, add_temp=False):
        """
        Method to predict on a given dataset using a fixed model

        PARAMETERS
        ----------
        net: PyTorch Model
            Network to test
        seq_loader: DataLoader
            Specific to the dataset used for test
        iteration: int
            Iteration used to display visualization
            Default: None
        get_quali: boolean
            If you want to save qualitative results
            Default: False
        add_temp: boolean
            Is the data are considered as a sequence
            Default: False
        """
        net.eval()
        transformations = get_transformations(self.transform_names, split='test',
                                              sizes=(self.w_size, self.h_size))
        ea_criterion = define_loss('elevation_azimuth', self.custom_loss, self.device)
        nb_losses = len(ea_criterion)
        running_losses = list()
        ea_running_losses = list()
        ea_running_global_losses = [list(), list()]
        ea_metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
                seq_name, seq = sequence_data
                path_to_frames = self.paths['dataset4d'] / seq_name[0]
                frame_dataloader = DataLoader(Dataset4dDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp),
                                              shuffle=False,
                                              batch_size=self.batch_size,
                                              num_workers=4)
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_ea = self.n_frames-1
                for j, frame in enumerate(frame_dataloader):
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
                    ea_outputs = net(ea_data, da_data, ed_data, er_data, ra_data)
                    ea_outputs = ea_outputs.to(self.device)

                    if get_quali:
                        quali_iter_ea = get_qualitatives(ea_outputs, ea_mask, self.paths,
                                                         seq_name, quali_iter_ea, 'elevation_azimuth')

                    ea_metrics.add_batch(torch.argmax(ea_mask, axis=1).cpu(),
                                         torch.argmax(ea_outputs, axis=1).cpu())

                    ea_losses = [c(ea_outputs, torch.argmax(ea_mask, axis=1))
                                    for c in ea_criterion]
                    ea_loss = torch.mean(torch.stack(ea_losses))
                    loss = torch.mean(ea_loss)

                    running_losses.append(loss.data.cpu().numpy()[()])
                    ea_running_losses.append(ea_loss.data.cpu().numpy()[()])
                    ea_running_global_losses[0].append(ea_losses[0].data.cpu().numpy()[()])
                    # Case with both sDice and CE losses
                    if nb_losses > 1:
                        ea_running_global_losses[1].append(ea_losses[1].data.cpu().numpy()[()])

                    if iteration and i == rand_seq:
                        if j == rand_frame:
                            ea_pred_masks = torch.argmax(ea_outputs, axis=1)[:5]
                            ea_gt_masks = torch.argmax(ea_mask, axis=1)[:5]
                            ea_pred_grid = make_grid(transform_masks_viz(ea_pred_masks,
                                                                         self.nb_classes))
                            ea_gt_grid = make_grid(transform_masks_viz(ea_gt_masks,
                                                                       self.nb_classes))
                            self.visualizer.update_multi_img_masks(ea_pred_grid, ea_gt_grid,
                                                                   iteration)
            self.test_results = dict()
            self.test_results['elevation_azimuth'] = get_metrics(ea_metrics, np.mean(ea_running_losses),
                                                             [np.mean(sub_loss) for sub_loss
                                                              in ea_running_global_losses])

            self.test_results['global_acc'] = (1/2)*(self.test_results['elevation_azimuth']['acc'])
            self.test_results['global_prec'] = (1/2)*(self.test_results['elevation_azimuth']['prec'])
            self.test_results['global_dice'] = (1/2)*(self.test_results['elevation_azimuth']['dice'])

            ea_metrics.reset()

        return self.test_results

    def write_params(self, path):
        """Write quantitative results of the Test"""
        with open(path, 'w') as fp:
            json.dump(self.test_results, fp)

    def set_device(self, device):
        """Set device used for test (supported: 'cuda', 'cpu')"""
        self.device = device

    def set_annot_type(self, annot_type):
        """Set annotation type to test on (specific to DATASET4D)"""
        self.annot_type = annot_type
