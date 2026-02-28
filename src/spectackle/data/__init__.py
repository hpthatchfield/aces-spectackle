### spectackle.data: synthetic spectrum generator and dataset
from spectackle.data.generator import DEFAULT_GEN, _make_v_axis, generate_spectrum
from spectackle.data.dataset import BASE_CFG, SyntheticSpectraDataset, make_loaders

__all__ = ["DEFAULT_GEN", "_make_v_axis", "generate_spectrum", "BASE_CFG", "SyntheticSpectraDataset", "make_loaders"]
