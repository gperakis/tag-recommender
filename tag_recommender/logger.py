import logging
from logging import config as logging_config
from pathlib import Path

import yaml


# Define the health endpoint filter
class HealthEndpointFilter(logging.Filter):
    """Ignore all the 200 OK logs from health endpoint"""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/health") == -1


def initialize_logging(config_path: Path):
    """
    Setup logging, according to the configuration, in the given file.

    Parameters
    ----------
    config_path : str
        The path to the file containing the logging configuration

    Returns
    -------
    None
    """
    with open(config_path) as yaml_fh:
        config_description = yaml.safe_load(yaml_fh)
        logging_config.dictConfig(config_description)
        logging.getLogger("uvicorn.access").addFilter(HealthEndpointFilter())
