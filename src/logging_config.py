import logging

def setup_logging(log_path, level):
    logging.basicConfig(filename=log_path, level=level,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logger = logging.getLogger()
    return logger
