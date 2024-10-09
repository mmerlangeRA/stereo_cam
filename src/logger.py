import logging
import os
from logging.handlers import RotatingFileHandler
from src.utils.path_utils import get_root, is_debug_mode

log_dir = os.path.join(get_root(), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Full path to the log file
log_file='app.log'
log_path = os.path.join(log_dir, log_file)
drop_in_console = True #is_debug_mode


def get_logger(name:str, max_bytes: int = 1e6, backup_count: int = 5)->logging.Logger:
    """
    Set up a logger that writes logs to both a file in the specified log directory and the console.
    The file will rotate when it reaches a certain size.
    
    Parameters:.
    - name (str): file name sending message.
    - max_bytes (int): Maximum size of a log file in bytes before it is rotated.
    - backup_count (int): Maximum number of backup files to keep.
    
    Returns:
    - logger: Configured logger instance.
    """

    # Create a logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the minimum log level for the logger

    # Create a rotating file handler for logging to a file with rotation
    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(logging.DEBUG)  # Set the log level for the file handler

    # Define the log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Attach the formatter to both handlers
    file_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)

    # Create a console handler for logging to the console (stdout)
    if drop_in_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Set the log level for the console handler
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Example usage


'''
log_file = 'app.log'  # Log file name
logger = setup_logger(log_dir, log_file)

# Log some messages to test the rotation
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')
'''