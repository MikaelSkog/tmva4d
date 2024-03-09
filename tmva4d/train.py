"""Main script to train a model"""
import argparse
import json
import torch.nn as nn
from tmva4d.utils.functions import count_params
from tmva4d.learners.initializer import Initializer
from tmva4d.learners.model import Model
from tmva4d.models import TMVA4D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.',
                        default='tmva4d/config_files/tmva4d.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    init = Initializer(cfg)
    data = init.get_data()
    
    net = TMVA4D(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

    Model(net, data).train(add_temp=True)

if __name__ == '__main__':
    main()
