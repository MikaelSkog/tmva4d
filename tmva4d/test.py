"""Main script to test a pretrained model"""
import argparse
import json
import torch
from torch.utils.data import DataLoader

from tmva4d.utils.paths import Paths
from tmva4d.utils.functions import count_params
from tmva4d.learners.tester import Tester
from tmva4d.models import TMVA4D
from tmva4d.loaders.dataset import Dataset4d
from tmva4d.loaders.dataloaders import SequenceDataset4dDataset

def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.',
                        default='config.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    paths = Paths().get()
    exp_name = cfg['name_exp'] + '_' + str(cfg['version'])
    path = paths['logs'] / cfg['dataset'] / cfg['model'] / exp_name
    model_path = path / 'results' / 'model.pt'
    test_results_path = path / 'results' / 'test_results.json'

    model = TMVA4D(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    print('Number of trainable parameters in the model: %s' % str(count_params(model)))
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    tester = Tester(cfg)
    data = Dataset4d()
    test = data.get('Test')
    testset = SequenceDataset4dDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    tester.set_annot_type(cfg['annot_type'])
    test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=True)
    tester.write_params(test_results_path)

if __name__ == '__main__':
    test_model()
