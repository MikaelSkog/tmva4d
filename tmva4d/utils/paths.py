"""Class to get global paths"""
from pathlib import Path
from tmva4d.utils import TMVA4D_HOME
from tmva4d.utils.configurable import Configurable


class Paths(Configurable):

    def __init__(self):
        self.config_path = TMVA4D_HOME / 'config_files' / 'config.ini'
        super().__init__(self.config_path)
        self.paths = dict()
        self._build()

    def _build(self):
        warehouse = Path(self.config['data']['warehouse'])
        self.paths['warehouse'] = warehouse
        self.paths['logs'] = Path(self.config['data']['logs'])
        self.paths['dataset4d'] = warehouse / 'Dataset4d'

    def get(self):
        return self.paths
