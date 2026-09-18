import sys, os

import logging
if __name__ == "__main__":
    logger = logging.getLogger("albatros_analysis.scripts.ionosonde")
else:
    logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.scripts.ionosonde import params

logger.debug("Test debug")
logger.info("Test info")
logger.warning("Test warning")