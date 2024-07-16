from modules.utils.file_system import ProjectPaths
from modules.utils.utilities import load_json_config

def initialize_config(config_file=ProjectPaths.config_file):
    def decorator(cls):
        cls.config = load_json_config(config_file)
        return cls
    return decorator

@initialize_config()
class Config:
    @staticmethod
    def get_config():
        return Config.config

    @staticmethod
    def get_data_config():
        return Config.config.data

    @staticmethod
    def get_model_config():
        return Config.config.model

    @staticmethod
    def get_condition_config():
        return Config.config.condition

    @staticmethod
    def get_training_config():
        return Config.config.training

    @staticmethod
    def get_logging_config():
        return Config.config.logging

    @staticmethod
    def get_profiler_config():
        return Config.config.profiler

    @staticmethod
    def get_telegram_config():
        return Config.config.telegram