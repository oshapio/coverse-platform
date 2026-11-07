"""Load one set of the benchmark: from the Hugging Face Hub, or from a local copy of the dataset folder."""
import os

DEFAULT_SOURCE = "Gigglingface/coverse-bench"
SETS = [f"{env}_d{depth}" for env in ("grid2d", "chars3d", "procthor3d", "ca1m3d") for depth in (1, 2, 3)]


def load_set(name, source=DEFAULT_SOURCE, images=True):
    """Rows of one set as dicts. `source` is a Hub dataset id or a local folder holding data/<set>.parquet.
    With images=True each row's "image" is {"bytes": <the image file>, "path": ...}; else the column is left out."""
    from datasets import Image, load_dataset

    if os.path.isdir(source):
        ds = load_dataset("parquet", data_files={name: os.path.join(source, "data", f"{name}.parquet")}, split=name)
    else:
        ds = load_dataset(source, split=name)
    ds = ds.cast_column("image", Image(decode=False)) if images else ds.remove_columns("image")
    return list(ds)
