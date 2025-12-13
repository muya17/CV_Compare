"""
Full unbiased comparison: HOG, YOLO, DeepLabV3, UNet on 4 datasets
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
import shutil
from ultralytics import YOLO
from torchvision import transforms, models

# ============================================================================
# DATASET SETUP
# ============================================================================

DATASET_CONFIG = {
    'CityScapes': r'C:\Users\User\Documents\CityU\Coursework\Year 3\Sem A\EE3070\data\DataSet\val\img',
    'NightTime': r'C:\Users\User\Documents\CityU\Coursework\Year 3\Sem A\EE3070\data\night_dataseet\nighttime_driving_dataset',
    'Underwater': r'C:\Users\User\Documents\CityU\Coursework\Year 3\Sem A\EE3070\data\under_water\TEST\images',
    'COCO': r'C:\Users\User\Documents\CityU\Coursework\Year 3\Sem A\EE3070\data\coco\coco'
}

TEST_DIR = 'test_datasets'
N_SAMPLES = 10

def sample_datasets():
    """Sample N images from each dataset"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    
    for dataset_name, dataset_path in DATASET_CONFIG.items():
        if not os.path.exists(dataset_path):
            print(f"⚠️  {dataset_name} not found")
            continue
        
        out_dir = os.path.join(TEST_DIR, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(Path(dataset_path).rglob(ext))
        
        if not images:
            print(f"⚠️  No images in {dataset_name}")
            continue
        
        np.random.seed(42)
        sampled = np.random.choice(images, min(N_SAMPLES, len(images)), replace=False)
        
        for i, src_img in enumerate(sampled, 1):
            # Normalize CityScapes PNGs: resize and convert to JPG to avoid stalls
            try:
                img = cv2.imread(str(src_img))
                if img is None:
                    continue
                h, w = img.shape[:2]
                max_side = max(h, w)
                if max_side > 1280:
                    scale = 1280.0 / max_side
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Always save as JPG for consistency and speed
                dst = os.path.join(out_dir, f'{dataset_name}_{i:02d}.jpg')
                cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            except Exception:
                # Fallback to raw copy
                dst = os.path.join(out_dir, f'{dataset_name}_{i:02d}{Path(src_img).suffix}')
                shutil.copy(src_img, dst)
        
        print(f"✓ Sampled {len(sampled)} from {dataset_name}")

# ============================================================================
# MODEL INFERENCE
# ============================================================================

def detect_hog(image_path):
    """HOG people detector"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    try:
        rects, _ = hog.detectMultiScale(img, winStride=(8, 8))
        return len(rects)
    except:
        return 0

def detect_yolo(yolo_model, image_path):
    """YOLO v8 detection"""
    try:
        results = yolo_model(image_path, verbose=False)
        detections = {}
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names.get(cls_id, f'class_{cls_id}')
                    detections[cls_name] = detections.get(cls_name, 0) + 1
        
        return detections
    except Exception as e:
        return {}

def detect_deeplab(model_dict, image_path):
    """DeepLabV3 segmentation"""
    if model_dict is None:
        return {}
    
    model = model_dict['model']
    device = model_dict['device']
    
    img = cv2.imread(image_path)
    if img is None:
        return {}
    
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)['out']
            seg_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        unique, counts = np.unique(seg_map, return_counts=True)
        class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor'
        ]
        
        detections = {}
        for cls_id, count in zip(unique, counts):
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            detections[cls_name] = int(count)
        
        # Return 0 if only background detected (no meaningful objects)
        if len(detections) == 1 and 'background' in detections:
            return {}
        
        return detections
    except Exception as e:
        return {}

def detect_unet(model_dict, image_path):
    """UNet segmentation"""
    if model_dict is None:
        return {}
    
    model = model_dict['model']
    device = model_dict['device']
    
    img = cv2.imread(image_path)
    if img is None:
        return {}
    
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            seg_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        unique, counts = np.unique(seg_map, return_counts=True)
        
        detections = {}
        for cls_id, count in zip(unique, counts):
            detections[f"class_{cls_id}"] = int(count)
        
        # Return 0 if untrained model produces noise (all same class or random output)
        if len(detections) <= 2:
            return {}
        
        return detections
    except Exception as e:
        return {}

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80, flush=True)
    print("UNBIASED MODEL COMPARISON - All 4 Models on All 4 Datasets", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Step 1: Sample
    print("📊 Sampling 10 images from each dataset...", flush=True)
    sample_datasets()
    
    # Step 2: Load models
    print("\n📦 Loading all 4 models...", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}", flush=True)
    
    # YOLO
    try:
        yolo = YOLO('yolov8n.pt')
        print("  ✓ YOLO v8", flush=True)
    except:
        print("  ✗ YOLO failed", flush=True)
        yolo = None
    
    # DeepLabV3
    try:
        deeplab_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        deeplab_model.to(device)
        deeplab_model.eval()
        deeplab = {'model': deeplab_model, 'device': device}
        print("  ✓ DeepLabV3", flush=True)
    except:
        print("  ✗ DeepLabV3 failed", flush=True)
        deeplab = None
    
    # UNet (clean version without camera)
    try:
        from unet_clean import SimpleUNet
        unet_model = SimpleUNet(in_channels=3, out_channels=21)
        unet_model.to(device)
        unet_model.eval()
        unet = {'model': unet_model, 'device': device}
        print("  ✓ UNet (clean)", flush=True)
    except Exception as e:
        print(f"  ✗ UNet failed: {str(e)[:50]}", flush=True)
        unet = None
    
    # Step 3: Run detection
    print("\n🔍 Running inference on all datasets...\n", flush=True)
    
    results = {}
    
    dataset_list = sorted(os.listdir(TEST_DIR))
    print(f"Datasets found: {dataset_list}", flush=True)

    # Process COCO, NightTime, Underwater first; CityScapes last
    preferred_order = ['COCO', 'NightTime', 'Underwater', 'CityScapes']
    ordered = [d for d in preferred_order if d in dataset_list] + [d for d in dataset_list if d not in preferred_order]
    
    for dataset_name in ordered:
        print(f"\n>>> Starting dataset: {dataset_name}", flush=True)
        dataset_path = os.path.join(TEST_DIR, dataset_name)
        if not os.path.isdir(dataset_path):
            print(f"  Skipping (not a directory)", flush=True)
            continue
        
        results[dataset_name] = {'HOG': {}, 'YOLO': {}, 'DeepLabV3': {}, 'UNet': {}}
        images = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"{'─'*80}", flush=True)
        print(f"📁 {dataset_name} ({len(images)} images)", flush=True)
        print(f"{'─'*80}", flush=True)
        
        try:
            for img_file in images:
                img_path = os.path.join(dataset_path, img_file)
                
                print(f"\n  {img_file}:", flush=True)
                
                # HOG
                try:
                    hog_count = detect_hog(img_path)
                    results[dataset_name]['HOG'][img_file] = hog_count
                    print(f"    HOG: {hog_count} people", flush=True)
                except Exception as e:
                    print(f"    HOG ERROR: {str(e)}", flush=True)
                    results[dataset_name]['HOG'][img_file] = 0
                
                # YOLO
                if yolo:
                    try:
                        yolo_dets = detect_yolo(yolo, img_path)
                        results[dataset_name]['YOLO'][img_file] = yolo_dets
                        print(f"    YOLO: {len(yolo_dets)} types - {yolo_dets if yolo_dets else 'None'}", flush=True)
                    except Exception as e:
                        print(f"    YOLO ERROR: {str(e)}", flush=True)
                        results[dataset_name]['YOLO'][img_file] = {}
                
                # DeepLabV3
                if deeplab:
                    try:
                        print(f"    Running DeepLabV3...", flush=True)
                        deeplab_dets = detect_deeplab(deeplab, img_path)
                        results[dataset_name]['DeepLabV3'][img_file] = deeplab_dets
                        if deeplab_dets:
                            top3 = sorted(deeplab_dets.items(), key=lambda x: x[1], reverse=True)[:3]
                            print(f"    DeepLabV3: {top3}", flush=True)
                        else:
                            print(f"    DeepLabV3: No output", flush=True)
                    except Exception as e:
                        print(f"    DeepLabV3 ERROR: {str(e)}", flush=True)
                        results[dataset_name]['DeepLabV3'][img_file] = {}
                
                # UNet
                if unet:
                    try:
                        print(f"    Running UNet...", flush=True)
                        unet_dets = detect_unet(unet, img_path)
                        results[dataset_name]['UNet'][img_file] = unet_dets
                        if unet_dets:
                            classes = sorted(unet_dets.keys())
                            print(f"    UNet: Classes {classes}", flush=True)
                        else:
                            print(f"    UNet: No output", flush=True)
                    except Exception as e:
                        print(f"    UNet ERROR: {str(e)}", flush=True)
                        results[dataset_name]['UNet'][img_file] = {}
                
                # Save progress after EACH image
                with open('unbiased_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"    ✓ Saved", flush=True)
        
        except Exception as e:
            print(f"\n⚠️  Error processing {dataset_name}: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            continue
        
        # Save progress after each dataset
        with open('unbiased_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved progress for {dataset_name}", flush=True)
        print(f">>> Finished dataset: {dataset_name}\n", flush=True)
    
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    for dataset_name in sorted(results.keys()):
        print(f"\n📊 {dataset_name}:")
        
        # HOG
        hog_with_det = sum(1 for v in results[dataset_name]['HOG'].values() if v > 0)
        print(f"  HOG: {hog_with_det}/10 images had people")
        
        # YOLO
        yolo_with_det = sum(1 for v in results[dataset_name]['YOLO'].values() if v)
        print(f"  YOLO: {yolo_with_det}/10 images had objects")
        
        # DeepLabV3
        deeplab_with_det = sum(1 for v in results[dataset_name]['DeepLabV3'].values() if v)
        print(f"  DeepLabV3: {deeplab_with_det}/10 images segmented")
        
        # UNet
        unet_with_det = sum(1 for v in results[dataset_name]['UNet'].values() if v)
        print(f"  UNet: {unet_with_det}/10 images detected")
    
    # Final save
    with open('unbiased_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Full results saved to unbiased_results.json")

if __name__ == '__main__':
    main()
