import logging
import os
import os.path


def setup_logging(verbose, logfile):

    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = logging.getLogger("train_model")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    preprocessing_logger = logging.getLogger("sleep_models.preprocessing")
    logger.setLevel(verbose)
    preprocessing_logger.setLevel(verbose)

    handler = logging.FileHandler(mode="w", filename=logfile)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    preprocessing_logger.addHandler(handler)

    return logger, preprocessing_logger
