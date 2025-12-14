"""
Sample from 4 real datasets and organize into test folders
Datasets:
  1. CityScapes - normal driving
  2. NightTime - night driving
  3. Underwater - underwater imagery
  4. COCO - general objects
"""

import shutil
import argparse
from pathlib import Path


def gather_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".JPG")):
    """Recursively find all image files"""
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def copy_n_images(src_dir, target_dir, prefix, n=10):
    """Copy first N images from src to target, renaming to prefix_01.ext, etc"""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear old files
    for p in target_dir.glob("*"):
        if p.is_file():
            p.unlink(missing_ok=True)
    
    images = gather_images(Path(src_dir))
    if not images:
        print(f"  ⚠️  No images found in {src_dir}")
        return 0
    
    copied = 0
    for i, src_path in enumerate(images[:n], 1):
        out_path = target_dir / f"{prefix}_{i:02d}{src_path.suffix.lower()}"
        shutil.copy2(src_path, out_path)
        copied += 1
    
    return copied


def main():
    parser = argparse.ArgumentParser(description="Sample N images from four datasets into test_datasets/")
    parser.add_argument("--city", default="C:/Users/User/Documents/CityU/Coursework/Year 3/Sem A/EE3070/data/DataSet/val/img", help="Path to CityScapes images")
    parser.add_argument("--night", default="C:/Users/User/Documents/CityU/Coursework/Year 3/Sem A/EE3070/data/night_dataseet/nighttime_driving_dataset", help="Path to NightTime images")
    parser.add_argument("--underwater", default="C:/Users/User/Documents/CityU/Coursework/Year 3/Sem A/EE3070/data/under_water/TEST/images", help="Path to Underwater images")
    parser.add_argument("--coco", default="C:/Users/User/Documents/CityU/Coursework/Year 3/Sem A/EE3070/data/coco/coco", help="Path to COCO images")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of images per dataset")
    args = parser.parse_args()

    print("="*70)
    print("SAMPLING FROM 4 DATASETS")
    print("="*70 + "\n")

    datasets = {
        "CityScapes": args.city,
        "NightTime": args.night,
        "Underwater": args.underwater,
        "COCO": args.coco,
    }

    test_base = Path("test_datasets")
    test_base.mkdir(exist_ok=True)

    for name, src_path in datasets.items():
        target_dir = test_base / name
        print(f"📂 {name}")
        print(f"   From: {src_path}")

        count = copy_n_images(src_path, target_dir, name.lower(), n=args.num)
        print(f"   ✓ Copied {count} images to {target_dir}\n")

    print("="*70)
    print(f"✅ All datasets sampled. Folder: {test_base}")
    print("="*70)
    print("\nNext: python core/unbiased_full.py")


if __name__ == "__main__":
    main()
