import ratspy as rp
import numpy as np
import pandas as pd
import pathlib


def fix_rp_kwargs(aug_name, rp_kwargs):
    """Convert string values in rp_kwargs to enums/tuples as needed for RATSpy."""

    rp_kwargs = rp_kwargs.copy()
    if aug_name == "AddNoise" and isinstance(rp_kwargs.get("noise_type", None), str):
        rp_kwargs["noise_type"] = getattr(rp.NoiseType, rp_kwargs["noise_type"])
    if aug_name == "AddNoise" and isinstance(rp_kwargs.get("bounds", None), list):
           rp_kwargs["bounds"] = tuple(rp_kwargs["bounds"])
    if aug_name == "Pool" and isinstance(rp_kwargs.get("kind", None), str):
        rp_kwargs["kind"] = getattr(rp.PoolingMethod, rp_kwargs["kind"])
    if aug_name.startswith("Convolve") and isinstance(
        rp_kwargs.get("window", None), str
    ):
        rp_kwargs["window"] = getattr(rp.ConvolveWindow, rp_kwargs["window"])
    if aug_name == "RandomTimeWarpAugmenter" and isinstance(
        rp_kwargs.get("speed_ratio_range", None), list
    ):
        rp_kwargs["speed_ratio_range"] = tuple(rp_kwargs["speed_ratio_range"])
    return rp_kwargs


def load_data(csv_path: pathlib.Path) -> (np.ndarray, list[str]):
    """Load dataset from CSV file."""

    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].astype(str).tolist()
    return X, y
