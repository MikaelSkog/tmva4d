"""Script to set the path to DATASET4D in the config.ini file"""
import argparse
from tmva4d.utils.configurable import Configurable
from tmva4d.utils import TMVA4D_HOME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings paths for training and testing.')
    parser.add_argument('--dataset4d', default='/datasets_local',
                        help='Path to the DATASET4D dataset.')
    parser.add_argument('--logs', default='/root/workspace/logs',
                        help='Path to the save the logs and models.')
    args = parser.parse_args()
    config_path = TMVA4D_HOME / 'config_files' / 'config.ini'
    configurable = Configurable(config_path)
    configurable.set('data', 'warehouse', args.dataset4d)
    configurable.set('data', 'logs', args.logs)
    with open(config_path, 'w') as fp:
        configurable.config.write(fp)
