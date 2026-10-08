"""Utility package for F2Mix.

Symbols are imported lazily (PEP 562) so that, e.g., the model and
copy-paste operators can be used without the data-processing dependencies
(h5py, nibabel, ...) being installed.
"""

_IMPORTS = {
    # Utils.Options
    'get_args': ('Utils.Options', 'get_args'),
    # Utils.common
    'setup_seed': ('Utils.common', 'setup_seed'),
    'setup_logging': ('Utils.common', 'setup_logging'),
    'make_run_dir': ('Utils.common', 'make_run_dir'),
    'save_checkpoint': ('Utils.common', 'save_checkpoint'),
    'load_checkpoint': ('Utils.common', 'load_checkpoint'),
    'load_weights': ('Utils.common', 'load_weights'),
    'strip_module_prefix': ('Utils.common', 'strip_module_prefix'),
    # Utils.datasets
    'HDF5VolumeDataset': ('Utils.datasets', 'HDF5VolumeDataset'),
    'FGAugmenter': ('Utils.datasets', 'FGAugmenter'),
    'weak_transform': ('Utils.datasets', 'weak_transform'),
    # Utils.fcp_ops
    'image_copy_paste': ('Utils.fcp_ops', 'image_copy_paste'),
    'feature_copy_paste': ('Utils.fcp_ops', 'feature_copy_paste'),
    'MutualInformation': ('Utils.fcp_ops', 'MutualInformation'),
    'NCC': ('Utils.fcp_ops', 'NCC'),
    'build_consistency_loss': ('Utils.fcp_ops', 'build_consistency_loss'),
    # Utils.metrics
    'compute_metrics': ('Utils.metrics', 'compute_metrics'),
    'plot_roc': ('Utils.metrics', 'plot_roc'),
    'CsvLogger': ('Utils.metrics', 'CsvLogger'),
    'save_predictions': ('Utils.metrics', 'save_predictions'),
}

__all__ = list(_IMPORTS)


def __getattr__(name):
    if name in _IMPORTS:
        module_name, attr = _IMPORTS[name]
        import importlib
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
