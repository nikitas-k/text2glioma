import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform, apply_orientation
from scipy.ndimage import generate_binary_structure, label as cc_label, convolve
from sklearn.utils.validation import check_random_state

from text2glioma.preprocessing.vasari_auto import get_vasari_features

# ---------- radiology-standard prompt composer (short + long) ----------
from typing import Optional, Dict, Any

VASARI_F1_REV = {1:"frontal lobe", 2:"temporal lobe", 3:"insula", 4:"parietal lobe",
                5:"occipital lobe", 6:"brainstem", 7:"corpus callosum", 8:"thalamus"}
VASARI_F2_REV = {1:"right", 2:"bilateral", 3:"left"}

# ---------- utils: LPS loader & volumes ----------
def load_nifti_LPS(path):
    img = nib.load(path)
    arr = img.get_fdata()
    codes = aff2axcodes(img.affine)
    if tuple(codes) != ('L','P','S'):
        orig = axcodes2ornt(codes)
        targ = axcodes2ornt(('L','P','S'))
        xform = ornt_transform(orig, targ)
        arr = apply_orientation(arr, xform)
        # reorder voxel sizes to match new axes
        z = img.header.get_zooms()[:3]
        order = xform[:, 0].astype(int)
        vox = tuple(z[i] for i in order)
    else:
        vox = img.header.get_zooms()[:3]
    return arr, vox  # arr shape (X,Y,Z) in LPS; vox = (vx,vy,vz) mm

def voxel_volume_ml(vox):  # mm^3 -> mL
    vx, vy, vz = map(float, vox)
    return (vx * vy * vz) / 1000.0

# ---------- enhancement / necrosis helpers ----------
def has_label(lab, label_id): return bool((lab == label_id).any())

def has_central_necrosis(lab, core_label=1, min_core_ml=5.0, vox=None):
    if vox is None: return has_label(lab, core_label)
    vol_ml = (lab == core_label).sum() * voxel_volume_ml(vox)
    return vol_ml >= float(min_core_ml)

# ---- RING ENHANCEMENT (ET surrounds core)
def detect_ring_enhancement(label_path, core_label=1, enh_label=3, shell_radius_vox=3, inner_clear_vox=1, min_ring_frac=0.30, min_touch_frac=0.50):
    lab, _ = load_nifti_LPS(label_path)
    core = (lab == core_label)
    enh  = (lab == enh_label)
    if not enh.any():  # no enhancing component
        return False, 0.0, 0.0

    # 3D morphology via convolution (structuring element cube)
    ksize = 2*shell_radius_vox + 1
    K = np.ones((ksize, ksize, ksize), dtype=np.uint8)

    # dilation/erosion (binary via conv thresholds)
    dil_core = convolve(core.astype(np.uint8), K, mode="constant", cval=0) > 0
    if inner_clear_vox > 0:
        kin = 2*inner_clear_vox + 1
        Kin = np.ones((kin, kin, kin), dtype=np.uint8)
        ero_core = convolve(core.astype(np.uint8), Kin, mode="constant", cval=0) == Kin.size
    else:
        ero_core = core

    shell = dil_core & (~ero_core)  # ring zone around core
    if shell.sum() == 0:
        return False, 0.0, 0.0

    enh_in_shell = (enh & shell).sum()
    ring_frac_shell = float(enh_in_shell) / float(shell.sum())       # how much of shell is enhancing
    # also require most ET to be near core
    enh_touch_core = (enh & dil_core).sum()
    touch_frac = float(enh_touch_core) / float(enh.sum())

    is_ring = (ring_frac_shell >= min_ring_frac) and (touch_frac >= min_touch_frac)
    return bool(is_ring), float(ring_frac_shell), float(touch_frac)

# ---------- edema severity (label 2) ----------
def edema_severity_from_labels(label_path, edema_label=2, brain_volume_ml=None):
    lab, vox = load_nifti_LPS(label_path)
    ed_vox = int((lab == edema_label).sum())
    ed_ml  = ed_vox * voxel_volume_ml(vox)

    # If you have brain volume, use fractions; else absolute ml thresholds.
    if brain_volume_ml is not None and brain_volume_ml > 0:
        frac = ed_ml / brain_volume_ml
        if   frac >= 0.20: sev = "severe"
        elif frac >= 0.10: sev = "moderate"
        elif frac >= 0.03: sev = "mild"
        else:              sev = "none"
        return sev, ed_ml, frac
    else:
        # Absolute ml heuristic (tune to cohort)
        if   ed_ml >= 100.0: sev = "severe"
        elif ed_ml >= 50.0:  sev = "moderate"
        elif ed_ml >= 20.0:  sev = "mild"
        else:                sev = "none"
        return sev, ed_ml, None

# Optional: rough brain volume estimate (if no brain mask) from T2/FLAIR by simple threshold
def estimate_brain_vol(image_path, lower_pct=15, upper_pct=99):
    img, vox = load_nifti_LPS(image_path)
    prc = np.percentile(img, [lower_pct, upper_pct])
    brain = (img > prc[0]) & (img < prc[1])  # crude intracranial proxy
    return brain.sum() * voxel_volume_ml(vox)

# ---------- mass effect severity (uses MLS + lesion volume) ----------
def lesion_volume_ml_from_labels(label_path, lesion_labels=(1,2,4)):
    lab, vox = load_nifti_LPS(label_path)
    lesion = np.isin(lab, lesion_labels)
    return lesion.sum() * voxel_volume_ml(vox)

def mass_effect_severity(mls_mm: float, lesion_ml: float):
    # Heuristic bins; adjust to your cohort/radiologist feedback
    if (mls_mm >= 10.0) or (lesion_ml >= 120.0 and mls_mm >= 5.0):
        return "severe"
    if (5.0 <= mls_mm < 10.0) or (lesion_ml >= 70.0):
        return "moderate"
    if (3.0 <= mls_mm < 5.0) or (lesion_ml >= 30.0):
        return "mild"
    return "none"

# ---------- location (reuse your fixed version) ----------
def location_from_label_LPS(label_path, nonenh_label=1, enh_label=3, bilateral_frac=0.10, midline_mm=5.0, vx_vol_ml=1.0):
    lab, vox = load_nifti_LPS(label_path)
    mask = (lab == enh_label) & (lab == nonenh_label)
    if not mask.any():
        tumour = lab == 3 # use edema instead
        if not tumour.any(): return None, None, None
        mask = tumour

    # --- multiple lesion check ---
    struct24 = generate_binary_structure(rank=3, connectivity=3)  # 3D, includes faces+edges+corners
    cc, num = cc_label(mask, structure=struct24)
    lesion_vols = [(cc == i).sum() * vx_vol_ml for i in range(1, num + 1)]
    total_vol_ml = sum(lesion_vols)

    if num > 1:
        lesion_text = f"{num} separate lesions totalling {total_vol_ml:.1f} mL"
    else:
        lesion_text = f"a lesion measuring {total_vol_ml:.1f} mL"
        
    X, Y, Z = mask.shape
    x_mid = (X - 1) / 2.0
    left_vox  = mask[int(np.floor(x_mid))+1:, :, :].sum()
    right_vox = mask[:int(np.ceil(x_mid)),  :, :].sum()
    vox_total = mask.sum()
    lf = float(left_vox) / float(vox_total); rf = float(right_vox) / float(vox_total)
    idxs = np.argwhere(mask); com = idxs.mean(0)
    x_dist_mm = abs(com[0] - x_mid) * float(voxel_volume_ml((vox[0],1,1)) * 1000)  # simplify to use vx; next line cleaner:
    x_dist_mm = abs(com[0] - x_mid) * float(vox[0])
    if lf > bilateral_frac and rf > bilateral_frac: laterality = "bilateral"
    elif x_dist_mm <= midline_mm:                   laterality = "midline"
    else:                                           laterality = "left" if lf >= rf else "right"
    y_frac = float(com[1] / max(Y - 1, 1)); z_frac = float(com[2] / max(Z - 1, 1))
    if   y_frac <= 1/3:   lobe = "frontal"
    elif y_frac >= 2/3:   lobe = "occipital"
    else:                 lobe = "temporal" if z_frac < 0.45 else "parietal"
    if laterality == "bilateral": return f"in the bilateral {lobe} lobes", num, lesion_text
    if laterality == "midline":   return f"in the midline {lobe} region", num, lesion_text
    return f"in the {laterality} {lobe} lobe", num, lesion_text

# ---------- midline shift (reuse your symmetry-based estimator) ----------
def estimate_midline_shift_mm(image_path, label_path, tumour_label_any=(1,2,4), search_vox=15):
    img, vox = load_nifti_LPS(image_path)
    lab, _   = load_nifti_LPS(label_path)
    X, Y, Z = img.shape; vx = float(vox[0])
    tumour = np.isin(lab, tumour_label_any)
    if not tumour.any(): return 0.0, "none", None
    areas = tumour.sum(axis=(0,1)); z = int(np.argmax(areas))
    I = img[:, :, z].astype(np.float32)
    p1, p99 = np.percentile(I, [1, 99]); I = np.clip((I - p1) / (p99 - p1 + 1e-6), 0, 1)
    x_mid0 = (X - 1) / 2.0
    best_cc, best_cx = -1.0, x_mid0
    for cx in range(int(x_mid0 - search_vox), int(x_mid0 + search_vox) + 1):
        cx = int(np.clip(cx, 1, X-2))
        w = int(min(cx, X - cx)); 
        if w < 8: continue
        L = I[cx - w:cx, :]; R = I[cx:cx + w, :]
        Lf = L[::-1, :]
        L0 = Lf - Lf.mean(); R0 = R - R.mean()
        denom = (np.linalg.norm(L0) * np.linalg.norm(R0) + 1e-6)
        cc = float((L0 * R0).sum() / denom)
        if cc > best_cc: best_cc, best_cx = cc, cx
    delta_vox = best_cx - x_mid0
    mls_mm = abs(delta_vox) * vx
    direction = "leftward" if delta_vox > 0 else ("rightward" if delta_vox < 0 else "none")
    return float(mls_mm), direction, z


def _map_fraction_code(name: str, code: Optional[int]) -> Optional[str]:
    if code is None or (isinstance(code, float) and np.isnan(code)): return None
    # VASARI-auto encodes:
    # Enhancing (F5): 3:<=5%, 4:5-33%, 5:33-67%, 6:67-100%
    bins = {
        "F5": {3:"≤5%", 4:"5–33%", 5:"33–67%", 6:"67–100%"},
        "F6": {3:"≤5%", 4:"5–33%", 5:"33–67%", 6:"67–95%", 7:"95–99.5%", 8:">99.5%"},
        "F7": {2:"none", 3:"minor", 4:"moderate", 5:"extensive"},
        "F11":{3:"thin/irregular rim", 4:"thick rim", 5:"thick rim without nCET"},
        "F14":{2:"none", 3:"mild", 4:"moderate", 5:"extensive"},
    }
    return bins.get(name, {}).get(int(code))

def _safe_int(v):
    try:
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else int(v)
    except Exception:
        return None

def _try_get_vasari(file_path: str, atlases_dir: Optional[str], **kwargs) -> Dict[str, Any]:
    out = {}
    if not atlases_dir:
        return out
    try:
        # assume get_vasari_features is imported from vasari-auto
        df = get_vasari_features(file_path, atlases=atlases_dir, **kwargs)
        row = df.iloc[0].to_dict()
        out["F1_loc_code"]  = _safe_int(row.get("F1 Tumour Location"))
        out["F2_side_code"] = _safe_int(row.get("F2 Side of Tumour Epicenter"))
        out["F4_quality"]   = _safe_int(row.get("F4 Enhancement Quality"))
        out["F5_enh"]       = _safe_int(row.get("F5 Proportion Enhancing"))
        out["F6_nCET"]      = _safe_int(row.get("F6 Proportion nCET"))
        out["F7_nec"]       = _safe_int(row.get("F7 Proportion Necrosis"))
        out["F11_rim"]      = _safe_int(row.get("F11 Thickness of enhancing margin"))
        out["F14_edema"]    = _safe_int(row.get("F14 Proportion of Oedema"))
        out["F19_epend"]    = _safe_int(row.get("F19 Ependymal Invasion"))    # 2 yes, 1 no
        out["F20_cortex"]   = _safe_int(row.get("F20 Cortical involvement"))   # 2 yes, 1 no
        out["F21_deepwm"]   = _safe_int(row.get("F21 Deep WM invasion"))       # 2 yes, 1 no
        out["F22_nCET_mid"] = _safe_int(row.get("F22 nCET Crosses Midline"))   # 3 yes, 2 no
        out["F23_CET_mid"]  = _safe_int(row.get("F23 CET Crosses midline"))    # 3 yes, 2 no
        out["F24_sats"]     = _safe_int(row.get("F24 satellites"))             # 2 yes, 1 no
        out["F9_multi"]     = _safe_int(row.get("F9 Multifocal or Multicentric"))
    except Exception:
        # If VASARI fails (missing atlases, etc.), return empty → fallback to heuristics only
        return {}
    return out

def compose_radiology_prompts(
    image_path: str,
    label_path: str,
    atlas_dir: Optional[str] = None,
    enhancing_label: int = 3,
    nonenhancing_label:  int = 1,
    edema_label:int = 2,
    z_dim: int = -1,
    cf: int = 1,
    t_ependymal: int = 5000,
    t_wm: int = 100,
    resolution: int = 1,
    midline_thresh: int = 5,
    enh_quality_thresh: int = 15,
    cyst_thresh: int = 50,
    cortical_thresh: int = 1000,
    focus_thresh: int = 30000,
    num_components_bin_thresh: int = 10,
    num_components_cet_thresh: int = 15,
    include_diag: str = "",
    shuffle_order: bool = True,
    seed: int | np.random.RandomState | None = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "short":  <one-line, Impression-style>,
          "long":   <Findings + Impression narrative>,
          "facts":  <dict of numeric/boolean facts for logging/debug>
        }
    """
    config = {
        "enhancing_label": enhancing_label,
        "nonenhancing_label": nonenhancing_label,
        "edema_label": edema_label,
        "z_dim": z_dim,
        "cf": cf,
        "t_ependymal": t_ependymal,
        "t_wm": t_wm,
        "resolution": resolution,
        "midline_thresh": midline_thresh,
        "enh_quality_thresh": enh_quality_thresh,
        "cyst_thresh": cyst_thresh,
        "cortical_thresh": cortical_thresh,
        "focus_thresh": focus_thresh,
        "num_components_bin_thresh": num_components_bin_thresh,
        "num_components_cet_thresh": num_components_cet_thresh,
        "verbose": verbose,
    }
    if seed:
        rng = check_random_state(seed)
    # ---------- optional VASARI overlay ----------
    va = _try_get_vasari(label_path, atlas_dir, **config)
    # derive human-readable bits
    va_loc = VASARI_F1_REV.get(va.get("F1_loc_code")) #if va.get("F1_loc_code") else None
    va_side = VASARI_F2_REV.get(va.get("F2_side_code")) #if va.get("F2_side_code") else None
    va_enh_bins = _map_fraction_code("F5", va.get("F5_enh"))
    va_nCET_bins= _map_fraction_code("F6", va.get("F6_nCET"))
    va_necros   = _map_fraction_code("F7", va.get("F7_nec"))
    va_rim      = _map_fraction_code("F11", va.get("F11_rim"))
    va_oedema   = _map_fraction_code("F14", va.get("F14_edema"))

    brain_ml = estimate_brain_vol(image_path, lower_pct=0, upper_pct=100)
    les_ml = lesion_volume_ml_from_labels(label_path, lesion_labels=(nonenhancing_label, edema_label, enhancing_label))
    mls_mm, mls_dir, mls_slice = estimate_midline_shift_mm(image_path, label_path, tumour_label_any=(nonenhancing_label, edema_label, enhancing_label), search_vox=15)
    me_sev = mass_effect_severity(mls_mm, les_ml)
    ed_sev, ed_ml, ed_frac = edema_severity_from_labels(label_path, edema_label=edema_label, brain_volume_ml=brain_ml)

    if va is None:
        # ---------- base facts from your existing helpers ----------
        # enhancement / ring
        is_ring, ring_shell_frac, touch_frac = detect_ring_enhancement(
            label_path, core_label=nonenhancing_label, enh_label=enhancing_label,
            shell_radius_vox=3, inner_clear_vox=1, min_ring_frac=0.30, min_touch_frac=0.50
        )
        lab, vox = load_nifti_LPS(label_path)
        nec = has_central_necrosis(lab, core_label=nonenhancing_label, min_core_ml=5.0, vox=vox)
    
        brain_ml = estimate_brain_vol(image_path, lower_pct=0, upper_pct=100)
        
    
    
        
    
        loc_text, n_cc, les_text = location_from_label_LPS(label_path, nonenh_label=nonenhancing_label,
                                                           enh_label=enhancing_label, bilateral_frac=0.10,
                                                           midline_mm=5.0, vx_vol_ml=voxel_volume_ml(vox))

    # ---------- phrase building (controlled vocabulary first, then heuristics) ----------
    # Enhancement phrase
    # if (lab == enh_label).any():
    #     enh_phrase = "ring enhancement" if is_ring else "solid enhancement"
    # else:
    #     enh_phrase = "no measurable enhancement"

    # Location/laterality
    loc_phrase = None
    if va_loc and va_side and va_side in ("left","right","bilateral"):
        if va_side == "bilateral":
            loc_phrase = f"bilateral {va_loc}s"
        else:
            loc_phrase = f"{va_side}-sided {va_loc}"

    # Invasion/midline
    cross_midline = (va.get("F23_CET_mid") == 3) or (va.get("F22_nCET_mid") == 3)
    ependymal_inv = (va.get("F19_epend") == 2)
    cortical_inv  = (va.get("F20_cortex") == 2)
    deepwm_inv    = (va.get("F21_deepwm") == 2)
    satellites    = (va.get("F24_sats") == 2)
    multifocal    = (va.get("F9_multi") == 2)

    inv_bits = []
    if cross_midline: inv_bits.append("crosses the midline")
    if ependymal_inv: inv_bits.append("ependymal involvement")
    if deepwm_inv:    inv_bits.append("deep white matter invasion")
    if cortical_inv:  inv_bits.append("cortical involvement")
    if satellites:    inv_bits.append("satellite lesions")
    if multifocal:    inv_bits.append("multifocal")

    # Size adjectives by brain fraction (coarse but radiology-friendly)
    size_adj = "small"
    if les_ml and brain_ml:
        frac = les_ml / max(brain_ml, 1e-6)
        if   frac >= 0.10: size_adj = "large"
        elif frac >= 0.05: size_adj = "moderate"
    else:
        if   les_ml >= 120: size_adj = "large"
        elif les_ml >= 50:  size_adj = "moderate"

    # Mass effect string
    me_bits = []
    if me_sev != "none": me_bits.append(f"{me_sev} mass effect")
    if mls_mm >= 3 and mls_dir != "none": me_bits.append(f"{mls_mm:.0f} mm {mls_dir} midline shift")
    me_phrase = ", ".join(me_bits) if me_bits else None

    # Edema string (prefer VASARI bin if present)
    edema_phrase = None
    if va_oedema:
        edema_phrase = f"{va_oedema} vasogenic edema"
        if ed_ml is not None: edema_phrase += f" (~{ed_ml:.0f} mL)"
    # elif ed_sev != "none":
    #     edema_phrase = f"{ed_sev} vasogenic edema"
    #     if ed_ml is not None: edema_phrase += f" (~{ed_ml:.0f} mL)"

    # Necrosis phrase (prefer VASARI)
    nec_phrase = None
    if va_necros and va_necros != "none":
        nec_phrase = "central necrosis"
    # elif bool(nec):
    #     nec_phrase = "central necrosis"

    # Rim thickness (VASARI)
    rim_phrase = None
    if va_rim: rim_phrase = va_rim

    # Enhancement extent (VASARI)
    enh_extent_phrase = None
    if va_enh_bins: enh_extent_phrase = f"enhancing component {va_enh_bins}"

    # ---------- SHORT (Impression-style, one sentence) ----------
    short_bits = []
    # order: lesion location -> size -> enhancement/rim/necrosis -> edema -> mass effect -> invasion/midline -> multiplicity -> optional dx
    if loc_phrase and size_adj: short_bits.append(f"{loc_phrase} {size_adj}-sized mass")
    #short_bits.append(f"{size_adj} lesion")
    if rim_phrase: short_bits.append(f"contrast enhancement with {rim_phrase}")
    else:
        short_bits.append("non-enhancing")
    if nec_phrase: short_bits.append(nec_phrase)
    if edema_phrase: short_bits.append(f"{va_oedema} vasogenic edema") # remove mL
    if me_phrase: short_bits.append(me_phrase)
    if inv_bits: short_bits.append(", ".join(inv_bits))
    if include_diag: short_bits.append(include_diag)

    # ---------- LONG (Findings → Impression) ----------
    findings = []
    if loc_phrase: findings.append(f"Location: {loc_phrase}.")
    findings.append(f"Lesion size class: {size_adj}; volume ≈ {les_ml:.1f} mL.")
    if enh_extent_phrase: findings.append(f"Enhancement quality: {enh_extent_phrase}")
    if rim_phrase: findings.append(f"Enhancing rim: {rim_phrase}.")
    if nec_phrase: findings.append("Necrosis: present.")
    if edema_phrase: findings.append(f"Edema: {edema_phrase}.")
    if me_phrase: findings.append(f"Mass effect: {me_phrase}.")
    if inv_bits: findings.append("Invasion/extension: " + ", ".join(inv_bits) + ".")
    #if n_cc and les_text: findings.append(f"Multiplicity: {les_text}.")
    if include_diag: findings.append(f"Diagnosis: {include_diag}")

    if shuffle_order:
        rng.shuffle(short_bits)
        rng.shuffle(findings)
    
    short = ", ".join(short_bits)
    short = short[0].upper() + short[1:]  # capitalise

    long_text = (
        " ".join(findings)
    )

    return {
        "short": short,
        "long": long_text,
        # "facts": {
        #     "lesion_ml": float(les_ml), "brain_ml": float(brain_ml),
        #     "mls_mm": float(mls_mm), "mls_dir": mls_dir, "mass_effect": me_sev,
        #     "edema_ml": float(ed_ml) if ed_ml is not None else None, "edema_sev": ed_sev,
        #     "ring_enhancement": bool(is_ring), "enh_shell_frac": float(ring_shell_frac),
        #     "enh_touch_frac": float(touch_frac), "necrosis": bool(nec),
        #     "location_text": loc_phrase, "vasari": va
        
    }

