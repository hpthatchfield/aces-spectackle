### spectackle.data: synthetic spectrum generator and dataset
from spectackle.data.dataset import BASE_CFG, SyntheticSpectraDataset, make_loaders
from spectackle.data.generator import (
    DEFAULT_GEN,
    FWHM_TO_SIGMA_KMS,
    _make_v_axis,
    channel_width_kms,
    fwhm_kms_to_sigma_kms,
    generate_spectrum,
)
from spectackle.data.nlw_dataset import NLWSpectraDataset, make_nlw_loaders
from spectackle.data.preprocess import prepare_spectrum_input, valid_mask
from spectackle.data.nlw_generator import (
    NLW_BASE_CFG,
    NLW_GEN_DEFAULT,
    build_nlw_base_cfg,
    generate_nlw_spectrum,
    nlw_cfg_velocity_window,
)
from spectackle.data.mopra_dataset import MOPRASpectraDataset, make_mopra_loaders
from spectackle.data.mopra_finetune_dataset import make_mopra_finetune_loaders
from spectackle.data.mopra_generator import (
    MOPRA_BASE_CFG,
    MOPRA_GEN_DEFAULT,
    MOPRA_GEN_SCOUSE_DAT,
    build_mopra_synth_cfg,
    generate_mopra_spectrum,
)
from spectackle.data.mopra_scouse_labels import build_scouse_labeled_cache, load_scouse_labeled_cache
from spectackle.data.mopra_header import build_mopra_base_cfg, mopra_axis_from_fits
from spectackle.data.mopra_preprocess import NORM_MODES, prepare_mopra_input, valid_mask_mopra
from spectackle.data.aces_generator import (
    ACES_BASE_CFG,
    ACES_GEN_DEFAULT,
    ACES_GEN_SIMPLE_GLANCE,
    ACES_GEN_SIMPLE_SNR,
    ACES_N_CHANNELS,
    ACES_VRANGE,
    build_aces_synth_cfg,
    generate_aces_spectrum,
)
from spectackle.data.aces_dataset import ACESSpectraDataset, make_aces_loaders

__all__ = [
    "DEFAULT_GEN",
    "_make_v_axis",
    "FWHM_TO_SIGMA_KMS",
    "channel_width_kms",
    "fwhm_kms_to_sigma_kms",
    "generate_spectrum",
    "BASE_CFG",
    "SyntheticSpectraDataset",
    "make_loaders",
    "NLW_GEN_DEFAULT",
    "NLW_BASE_CFG",
    "build_nlw_base_cfg",
    "nlw_cfg_velocity_window",
    "generate_nlw_spectrum",
    "NLWSpectraDataset",
    "make_nlw_loaders",
    "valid_mask",
    "prepare_spectrum_input",
    "MOPRA_BASE_CFG",
    "MOPRA_GEN_DEFAULT",
    "MOPRA_GEN_SCOUSE_DAT",
    "build_mopra_base_cfg",
    "build_mopra_synth_cfg",
    "build_scouse_labeled_cache",
    "generate_mopra_spectrum",
    "make_mopra_finetune_loaders",
    "MOPRASpectraDataset",
    "make_mopra_loaders",
    "load_scouse_labeled_cache",
    "NORM_MODES",
    "prepare_mopra_input",
    "valid_mask_mopra",
    "mopra_axis_from_fits",
    "ACES_BASE_CFG",
    "ACES_GEN_DEFAULT",
    "ACES_GEN_SIMPLE_GLANCE",
    "ACES_GEN_SIMPLE_SNR",
    "ACES_N_CHANNELS",
    "ACES_VRANGE",
    "ACESSpectraDataset",
    "build_aces_synth_cfg",
    "generate_aces_spectrum",
    "make_aces_loaders",
]
