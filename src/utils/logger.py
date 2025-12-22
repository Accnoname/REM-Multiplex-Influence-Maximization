# TODO: Implement logic here
# src/utils/logger.py
import logging
import sys

def setup_logger(name="REM"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Tạo handler in ra màn hình console
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger