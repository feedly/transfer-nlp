"""
This class aims at using a pytorch loss function into ignite Metrics
"""

import logging

from ignite.metrics import Loss

from transfer_nlp.plugins.config import register_plugin

name = 'transfer_nlp.plugins.metrics'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)


@register_plugin
class LossMetric(Loss):
    """
    avoid name collision on batch size param of super class
    """

    def __init__(self, loss_fn):
        super().__init__(loss_fn)
