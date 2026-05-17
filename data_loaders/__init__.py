"""
Data loaders dispatch
---------------------
Single entry point for all data-loading. Picks between the local loader and
the imported loader based on RunConfig.dataset_mode.

Every loader returns:
    make_loaders: callable returning (train_loader, val_loader) for a trial
    data_info:    dict with input_dim / output_dim / input_channels / etc.
"""

from core.run_config import RunConfig


def load_data(cfg: RunConfig):
    # Language modeling on lyrics: dedicated loader (handles Word2Vec + MIDI)
    if cfg.data_type == "lyrics" or cfg.task_type == "language_modeling":
        from data_loaders.lyrics_loader import load_lyrics
        return load_lyrics(cfg)
    if cfg.dataset_mode == "local":
        from data_loaders.local_loader import load_local
        return load_local(cfg)
    if cfg.dataset_mode == "imported":
        from data_loaders.imported_loader import load_imported
        return load_imported(cfg)
    raise ValueError(f"Unknown dataset_mode '{cfg.dataset_mode}'.")
