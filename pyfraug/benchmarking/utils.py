def fix_pf_kwargs(aug_name, pf_kwargs):
    """Convert string values in pf_kwargs to enums/tuples as needed for PyFraug."""
    import pyfraug as pf
    pf_kwargs = pf_kwargs.copy()
    if aug_name == "AddNoise" and isinstance(pf_kwargs.get("noise_type", None), str):
        pf_kwargs["noise_type"] = getattr(pf.NoiseType, pf_kwargs["noise_type"])
    if aug_name == "Pool" and isinstance(pf_kwargs.get("kind", None), str):
        pf_kwargs["kind"] = getattr(pf.PoolingMethod, pf_kwargs["kind"])
    if aug_name.startswith("Convolve") and isinstance(pf_kwargs.get("window", None), str):
        pf_kwargs["window"] = getattr(pf.ConvolveWindow, pf_kwargs["window"])
    if aug_name == "RandomTimeWarpAugmenter" and isinstance(pf_kwargs.get("speed_ratio_range", None), list):
        pf_kwargs["speed_ratio_range"] = tuple(pf_kwargs["speed_ratio_range"])
    return pf_kwargs