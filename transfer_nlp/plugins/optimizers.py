"""
If you decide to implement your own optimizers, you should implement an optim class with the same interface than torch.optim schedulers.
There are already the most used pytorch schedulers included in the registry so you should be good to go, but you can still
implement yours and add the decorator @register_scheduler which is in transfer_nlp.plugins.registry.py
"""
