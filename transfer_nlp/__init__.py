import logging

logger = logging.getLogger(__name__)

try:
    from transfer_nlp.plugins import metrics, regularizers, helpers, trainers
    logger.debug("Using trainers with Torch utilities")
except ImportError as e:
    logger.debug("Using trainers without Torch utilities")
