# TODO: integrate this in project architecture
import os

class FileSystem:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.models_dir = os.path.join(base_dir, 'models')
        self.logs_dir = os.path.join(base_dir, 'logs')

    def create_dirs(self):
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def get_data_path(self, filename):
        return os.path.join(self.data_dir, filename)

    def get_model_path(self, filename):
        return os.path.join(self.models_dir, filename)

    def get_log_path(self, filename):
        return os.path.join(self.logs_dir, filename)
