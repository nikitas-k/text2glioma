"""text2glioma.validation — validation pipeline modules.

Submodules
----------
image_quality
    FID, MS-SSIM, pixel-level statistics (KS, CNR, SNR).
mask_fidelity
    Round-trip segmentation Dice / HD95, mask ablation.
text_alignment
    VASARI feature recovery, CLIP text–image score.
downstream_utility
    Classifier experiments across data regimes.
ablation
    Guidance scale sweeps, DDIM steps, dropout config generation.
diversity
    Nearest-neighbour memorisation check, intra-prompt diversity.
radiologist_eval
    Turing test preparation, quality rating forms, analysis.
"""

from text2glioma.validation.image_quality import run_image_quality
from text2glioma.validation.mask_fidelity import run_mask_fidelity
from text2glioma.validation.text_alignment import run_text_alignment
from text2glioma.validation.downstream_utility import run_downstream_grid
from text2glioma.validation.diversity import run_diversity
from text2glioma.validation.radiologist_eval import (
    prepare_turing_test,
    prepare_quality_rating,
    analyse_turing_test,
    analyse_quality_rating,
)

__all__ = [
    "run_image_quality",
    "run_mask_fidelity",
    "run_text_alignment",
    "run_downstream_grid",
    "run_diversity",
    "prepare_turing_test",
    "prepare_quality_rating",
    "analyse_turing_test",
    "analyse_quality_rating",
]
