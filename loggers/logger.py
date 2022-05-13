import os
import logging
import logging.config
from utils import utils


def setup_logging(save_dir, log_config='./loggers/logger_config.json'):
    """
    Setup logging configuration
    """
    config = utils.read_json(log_config)
    for _, handler in config['handlers'].items():
        if 'filename' in handler:
            handler['filename'] = os.path.join(save_dir, handler['filename'])

    logging.config.dictConfig(config)
