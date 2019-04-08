"""
If you decide to implement your own schedulers, you should implement an optim.lr_scheduler class with the same interface than optim.lr_scheduler schedulers.
There are already the most used pytorch optimizers included in the registry so you should be good to go, but you can still
implement yours and add the decorator @register_scheduler which is in transfer_nlp.plugins.registry.py
"""