import logging
 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # don't also hand messages to the root logger
 
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] line %(lineno)d in %(filename)s:\n\t%(message)s"
))
logger.addHandler(console_handler)