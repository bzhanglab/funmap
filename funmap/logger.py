import yaml
import logging
import logging.config
from pathlib import Path

from .saving import log_path

LOG_LEVEL = logging.INFO

def setup_logging(run_config, log_config="logging.yml") -> None:
    """
    Setup ``logging.config``

    Parameters
    ----------
    run_config : str
        Path to configuration file for run

    log_config : str
        Path to configuration file for logging
    """
    log_config = Path(log_config)

    if not log_config.exists():
        log_folder = log_path(run_config)
        log_file_debug = log_folder / "debug.log"
        log_file_info = log_folder / "info.log"
        log_file_warning = log_folder / "warning.log"
        log_file_error = log_folder / "error.log"
        log_file_critical = log_folder / "critical.log"

        # Create separate log handlers for each log file
        handler_debug = logging.FileHandler(filename=str(log_file_debug))
        handler_info = logging.FileHandler(filename=str(log_file_info))
        handler_warning = logging.FileHandler(filename=str(log_file_warning))
        handler_error = logging.FileHandler(filename=str(log_file_error))
        handler_critical = logging.FileHandler(filename=str(log_file_critical))

        # Set log levels for each handler
        handler_debug.setLevel(logging.DEBUG)
        handler_info.setLevel(logging.INFO)
        handler_warning.setLevel(logging.WARNING)
        handler_error.setLevel(logging.ERROR)
        handler_critical.setLevel(logging.CRITICAL)

        # Create formatters (if needed)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler_debug.setFormatter(formatter)
        handler_info.setFormatter(formatter)
        handler_warning.setFormatter(formatter)
        handler_error.setFormatter(formatter)
        handler_critical.setFormatter(formatter)

        # Get the root logger
        logger = logging.getLogger()

        # Add the handlers to the root logger
        logger.addHandler(handler_debug)
        logger.addHandler(handler_info)
        logger.addHandler(handler_warning)
        logger.addHandler(handler_error)
        logger.addHandler(handler_critical)

        logger.setLevel(LOG_LEVEL)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Set the level for console output
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Log a warning message to the console
        logger.warning(f'"{log_config}" not found. Using basicConfig with custom log files.')
        return

    with open(log_config, "rt") as f:
        config = yaml.safe_load(f.read())

    # modify logging paths based on run config
    run_path = log_path(run_config)
    for _, handler in config["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(run_path / handler["filename"])

    logging.config.dictConfig(config)


def setup_logger(name):
    log = logging.getLogger(f'funmap.{name}')
    log.setLevel(LOG_LEVEL)
    return log
