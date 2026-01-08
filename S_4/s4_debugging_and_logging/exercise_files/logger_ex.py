# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.info("This is an info message")

# Use this instead
from loguru import logger
logger.info("This is an info message")

if debug:
    print(x.shape)