import os

class Config:
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        for key, value in config_dict.items():
            setattr(self, key, value)


class DefaultConfig(Config):
    def __init__(self):
        super().__init__(
            data_dir = './data/',
            R_home_path = os.environ.get('R_HOME', None),
            pyscenic_path = None,
        )
    
    