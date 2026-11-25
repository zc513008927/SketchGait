
import argparse
import logging
import os
import pickle
import concurrent.futures
from functools import partial
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. MAPPING CONFIGURATION
# ==========================================

# --- CIHP Parsing Config ---
CIHP_PALETTE_RGB = [
    [0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51],
    [255, 85, 0], [0, 0, 85], [0, 119, 221], [85, 85, 0], [0, 85, 85],
    [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255], [51, 170, 221],
    [0, 255, 255], [85, 255, 170], [170, 255, 85], [255, 255, 0], [255, 170, 0]
]

MAPPING_RULES_middle = {
    0: [0],  # Background
    2: [13], 3: [4], 4: [2], 5: [1],  # Head
    6: [5], 7: [6], 8: [7], 9: [10], 10: [11], 11: [12],  # Torso
    12: [9], 13: [8], 14: [16], 15: [17],  # Legs
    16: [14], 17: [15], 18: [3],  # Arms
    19: [18], 20: [19]  # Shoes
}

def build_lookup_table() -> Dict[Tuple[int, int, int], int]:
    color_to_id = {}
    for new_id, old_id_list in MAPPING_RULES_middle.items():
        for old_id in old_id_list:
            if old_id >= len(CIHP_PALETTE_RGB):
                continue
            rgb = CIHP_PALETTE_RGB[old_id]
            bgr_tuple = (rgb[2], rgb[1], rgb[0])
            color_to_id[bgr_tuple] = new_id
    return color_to_id


GLOBAL_COLOR_MAP = build_lookup_table()


def convert_rgb_to_label(img_bgr: np.ndarray) -> np.ndarray:
    """Parsing: BGR -> Label (0-20)"""
    h, w = img_bgr.shape[:2]
    new_mask = np.zeros((h, w), dtype=np.uint8)
    # strict matching
    for bgr_tuple, new_label_id in GLOBAL_COLOR_MAP.items():
        matches = np.all(img_bgr == bgr_tuple, axis=-1)
        new_mask[matches] = new_label_id
    return new_mask



# ==========================================
# Optional Skeleton Color Map (BGR exact match)
# ==========================================
SKELETON_COLOR_MAP = {
    (0, 255, 0): 23,    # left_arm (Green)
    (0, 0, 255): 24,    # right_arm (Red)
    (255, 255, 0): 25,  # left_forearm (Cyan)
    (0, 255, 255): 26,  # right_forearm (Yellow)
    (255, 0, 255): 27,  # left_thigh (Magenta)
    (128, 0, 0): 28,    # right_thigh (Dark Blue)
    (0, 128, 0): 29,    # left_calf (Dark Green)
    (0, 0, 128): 30     # right_calf (Dark Red)
}


def convert_edge_to_label_v2(img_bgr: np.ndarray) -> np.ndarray:
    """Edge Modality B: colored skeleton -> labels (23-30)"""
    h, w = img_bgr.shape[:2]
    final_label_mask = np.zeros((h, w), dtype=np.uint8)
    for color_tuple, label_id in SKELETON_COLOR_MAP.items():
        matches = np.all(img_bgr == color_tuple, axis=-1)
        final_label_mask[matches] = label_id
    return final_label_mask


# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def sanitize_filename(filename: str) -> str:
    return filename.translate(str.maketrans('<>:"/\\|?*', '_________'))


def get_image_list(dir_path: Path) -> List[Path]:
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(list(dir_path.glob(ext)))
    return sorted(images)


def img2array(img_path: Path, read_as_color: bool) -> np.ndarray:
    if read_as_color:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    return img


def iter_leaf_dirs_with_images(root_dir: Path):
    """
    Yield directories that contain at least one image file.
    This avoids collecting ALL image paths into memory.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                yield Path(dirpath)
                break


# ==========================================
# 3. CORE PROCESSING LOGIC
# ==========================================

def process_three_sequence(
    dir_a: Path,
    root_path_a: Path,
    root_path_b: Path,
    root_path_c: Path,
    output_path_a: Path,
    output_path_b: Path,
    output_path_c: Path,
    img_size: int = 64,
    verbose: bool = False,
    use_edge_v2: bool = False,
) -> None:
    """Align A (silhouette), B (parsing), C (grayscale) using only A's geometry."""
    try:
        relative_path = dir_a.relative_to(root_path_a)
        dir_b = root_path_b / relative_path
        dir_c = root_path_c / relative_path
        if not dir_b.is_dir() or not dir_c.is_dir():
            if verbose:
                logging.warning(f"Missing B/C sequence for {relative_path}")
            return

        sid, seq, view = (sanitize_filename(p) for p in relative_path.parts[-3:])

        # Match by file stem, rather than relying on sorted zip order.
        files_a = {p.stem: p for p in get_image_list(dir_a)}
        files_b = {p.stem: p for p in get_image_list(dir_b)}
        files_c = {p.stem: p for p in get_image_list(dir_c)}
        common_stems = sorted(files_a.keys() & files_b.keys() & files_c.keys())
        if not common_stems:
            if verbose:
                logging.warning(f"No common frames for {relative_path}")
            return

        processed_frames_a = []
        processed_frames_b = []
        processed_frames_c = []

        for stem in common_stems:
            try:
                # A is the reference silhouette.
                raw_a = img2array(files_a[stem], read_as_color=False)
                img_a_sil = (raw_a > 0).astype(np.uint8) * 255

                # B is semantic parsing; retain the original color-to-label mapping.
                raw_b = img2array(files_b[stem], read_as_color=True)
                img_b_label = convert_rgb_to_label(raw_b)

                # C follows the same grayscale + binarization convention as A.
                raw_c = img2array(files_c[stem], read_as_color=False)
                img_c_sil = (raw_c > 0).astype(np.uint8) * 255

                if img_a_sil.shape != img_b_label.shape or img_a_sil.shape != img_c_sil.shape:
                    raise ValueError(f"Shape mismatch at {relative_path}/{stem}")

                if img_a_sil.sum() <= 10000:
                    continue

                fg_rows = np.flatnonzero(np.any(img_a_sil > 0, axis=1))
                if fg_rows.size == 0:
                    continue
                y_top, y_btm = int(fg_rows[0]), int(fg_rows[-1])

                crop_a = img_a_sil[y_top:y_btm + 1, :]
                crop_b = img_b_label[y_top:y_btm + 1, :]
                crop_c = img_c_sil[y_top:y_btm + 1, :]
                if crop_a.shape[0] == 0:
                    continue

                ratio = crop_a.shape[1] / crop_a.shape[0]
                target_width = max(1, int(img_size * ratio))
                size = (target_width, img_size)
                resized_a = cv2.resize(crop_a, size, interpolation=cv2.INTER_CUBIC)
                resized_b = cv2.resize(crop_b, size, interpolation=cv2.INTER_NEAREST)
                resized_c = cv2.resize(crop_c, size, interpolation=cv2.INTER_NEAREST)

                x_csum = resized_a.sum(axis=0).cumsum()
                total_mass = resized_a.sum()
                if total_mass <= 0:
                    continue
                # Same horizontal mass-median rule as the original script.
                x_center = int(np.searchsorted(x_csum, total_mass / 2, side='right'))
                if x_center >= resized_a.shape[1]:
                    continue

                half_width = img_size // 2
                left_bound = x_center - half_width
                right_bound = left_bound + img_size

                def apply_pad(img_arr, l_bound, r_bound):
                    h2, w2 = img_arr.shape
                    pad_l = max(0, -l_bound)
                    pad_r = max(0, r_bound - w2)
                    if pad_l or pad_r:
                        img_arr = np.pad(img_arr, ((0, 0), (pad_l, pad_r)), mode='constant')
                        l_bound += pad_l
                        r_bound += pad_l
                    return img_arr[:, l_bound:r_bound]

                final_a = apply_pad(resized_a, left_bound, right_bound)
                final_b = apply_pad(resized_b, left_bound, right_bound)
                final_c = apply_pad(resized_c, left_bound, right_bound)
                if any(frame.shape != (img_size, img_size) for frame in (final_a, final_b, final_c)):
                    raise ValueError(f"Invalid output shape at {relative_path}/{stem}")

                # Append all three together so temporal indexing remains aligned.
                processed_frames_a.append(final_a)
                processed_frames_b.append(final_b)
                processed_frames_c.append(final_c)
            except Exception as exc:
                logging.warning(f"Skipped frame {relative_path}/{stem}: {exc}")

        if processed_frames_a:
            for output_root, frames in (
                (output_path_a, processed_frames_a),
                (output_path_b, processed_frames_b),
                (output_path_c, processed_frames_c),
            ):
                out_dir = output_root / relative_path
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / f'{view}.pkl', 'wb') as f:
                    pickle.dump(np.stack(frames).astype(np.uint8), f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                logging.info(f"Saved A+B+C: {len(processed_frames_a)} frames for {(sid, seq, view)}")
    except Exception as exc:
        logging.exception(f"Error processing {dir_a}: {exc}")


# ==========================================
# 4. ENTRY POINT (ID-by-ID + threading inside ID)
# ==========================================

def pretreat_by_id(
    input_a: Path,
    output_a: Path,
    input_b: Path,
    output_b: Path,
    input_c: Path,
    output_c: Path,
    img_size: int = 64,
    workers: int = 32,
    verbose: bool = False,
    start_index: int = 0,
    use_edge_v2: bool = False,
):
    """
    Optimized:
    - Scan ID folders only (first level under input_a)
    - For each ID: find leaf dirs with images (os.walk)
    - Thread pool processes leaf dirs within this ID only
    """
    logging.info(f'Input A (Silhouette): {input_a}')
    logging.info(f'Input B (Parsing): {input_b}')
    logging.info(f'Input C (Sketch): {input_c}')

    id_folders = sorted([p for p in input_a.iterdir() if p.is_dir()])
    if not id_folders:
        print("No ID folders found in Input A!")
        return

    total_ids = len(id_folders)
    print("=" * 60)
    print("Three Modality Processing: ID-by-ID Threading Edition")
    print(f"Workers: {workers}")
    print(f"Total IDs: {total_ids}")
    print("=" * 60)

    # fixed-args function
    p_process = partial(
        process_three_sequence,
        root_path_a=input_a,
        root_path_b=input_b,
        root_path_c=input_c,
        output_path_a=output_a,
        output_path_b=output_b,
        output_path_c=output_c,
        img_size=img_size,
        verbose=verbose,
        use_edge_v2=use_edge_v2
    )

    for idx, id_path in enumerate(id_folders):
        if idx < start_index:
            continue

        # Only scan current ID
        seq_dirs = list(iter_leaf_dirs_with_images(id_path))
        # De-duplicate dirs (os.walk may yield duplicates in rare cases)
        seq_dirs = sorted(set(seq_dirs))

        if not seq_dirs:
            continue

        print(f"\n[{idx + 1}/{total_ids}] ID={id_path.name}  sequences={len(seq_dirs)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(p_process, d): d for d in seq_dirs}

            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(seq_dirs),
                desc=f"ID {id_path.name}",
                unit="seq",
                leave=False
            ):
                if verbose:
                    try:
                        fut.result()
                    except Exception as e:
                        logging.error(f"Task failed: {e}")

        print(f"ID {id_path.name} done.")

    print("\nAll tasks completed.")


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-a',
        default=r'silhouette path',
        type=str
    )
    parser.add_argument(
        '--output-a',
        default=r'',
        type=str
    )
    parser.add_argument(
        '--input-b',
        default=r'parsing path',
        type=str
    )
    parser.add_argument(
        '--output-b',
        default=r'',
        type=str
    )

    parser.add_argument(
        '--input-c',
        default=r'sketch path',
        type=str,)
    parser.add_argument(
        '--output-c',
        default=r'',
        type=str)

    parser.add_argument('--n_workers', default=8, type=int)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--start_index', default=0, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pretreat_by_id(
        input_a=Path(args.input_a),
        output_a=Path(args.output_a),
        input_b=Path(args.input_b),
        output_b=Path(args.output_b),
        input_c=Path(args.input_c),
        output_c=Path(args.output_c),
        img_size=args.img_size,
        workers=args.n_workers,
        verbose=args.verbose,
        start_index=args.start_index
    )
