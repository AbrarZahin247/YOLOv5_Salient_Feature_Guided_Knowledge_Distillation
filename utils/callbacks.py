# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Callback functions for YOLOv5 models
"""

import inspect


class Callbacks:
    """
    Handles all YOLOv5 callback functions
    """

    def __init__(self):
        # Define callback dictionary with all required hooks
        self._callbacks = {
            'on_pretrain_routine_start': [],
            'on_pretrain_routine_end': [],
            'on_train_start': [],
            'on_train_epoch_start': [],
            'on_train_batch_start': [],
            'optimizer_step': [],
            'on_before_zero_grad': [],
            'on_train_batch_end': [],
            'on_train_epoch_end': [],
            'on_val_start': [],
            'on_val_batch_start': [],
            'on_val_image_end': [],
            'on_val_batch_end': [],
            'on_val_end': [],
            'on_fit_epoch_end': [],
            'on_model_save': [],
            'on_train_end': [],
            'on_params_update': [],
            'check_dataset': [],
            'process_preds': [],
            'upload_dataset_to_wandb': [],  # <-- THE FINAL FIX
            'teardown': []}
        self.stop_training = False

    def register_action(self, hook, name='', callback=None):
        """
        Register a new action to a callback hook
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks.keys()}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """
        Returns all registered actions by hook
        """
        return self._callbacks if hook is None else self._callbacks.get(hook, [])

    def run(self, hook, *args, **kwargs):
        """
        Run all actions registered to a specific callback hook
        """
        for logger in self._callbacks[hook]:
            # Get logger arguments
            logger_args = inspect.getfullargspec(logger['callback']).args
            # Pass only accepted arguments
            kwargs_ = {k: v for k, v in kwargs.items() if k in logger_args}
            logger['callback'](*args, **kwargs_)
