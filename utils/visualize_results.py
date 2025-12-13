"""
Model Performance Comparison - Visual Summary
Shows detection rates across datasets and highlights underwater performance drop
"""
import json
import matplotlib.pyplot as plt
import numpy as np

with open('unbiased_results.json', 'r') as f:
    results = json.load(f)

# Calculate detection rates for each model per dataset
datasets = ['COCO', 'CityScapes', 'NightTime', 'Underwater']
models = ['HOG', 'YOLO', 'DeepLabV3', 'UNet']

detection_rates = {model: [] for model in models}

for dataset in datasets:
    for model in models:
        detections = results[dataset].get(model, {})
        
        if model == 'UNet':
            # UNet is useless - just show it's random
            detected = sum(1 for v in detections.values() if v)
            total = len(detections)
            rate = detected / total * 100 if total > 0 else 0
            detection_rates[model].append(rate)
        elif model == 'DeepLabV3':
            # Only count if it detected REAL objects (not just background)
            detected = 0
            for v in detections.values():
                if isinstance(v, dict):
                    # Check if it has classes OTHER than background
                    has_real_objects = any(cls != 'background' for cls in v.keys())
                    if has_real_objects:
                        detected += 1
            total = len(detections)
            rate = detected / total * 100 if total > 0 else 0
            detection_rates[model].append(rate)
        else:
            # Real models - count images with detections
            if model == 'HOG':
                detected = sum(1 for v in detections.values() if v > 0)
            else:
                detected = sum(1 for v in detections.values() if v)
            
            total = len(detections)
            rate = detected / total * 100 if total > 0 else 0
            detection_rates[model].append(rate)

# Create figure with better styling
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(datasets))
width = 0.2
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']

for i, model in enumerate(models):
    offset = (i - 1.5) * width
    if model == 'UNet':
        # Show UNet as a dashed line to indicate it's untrained/useless
        ax.plot(x + offset, detection_rates[model], 'o--', label=model + ' (untrained)', 
                linewidth=2, markersize=8, color=colors[i], alpha=0.5)
    else:
        ax.bar(x + offset, detection_rates[model], width, label=model, color=colors[i], alpha=0.8)

ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Detection Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Across Different Environments\n(Higher = Better)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotations
ax.text(0.5, 95, 'YOLO strongest', fontsize=10, ha='center', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
ax.text(2.5, 20, 'All models fail\nat night', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
ax.text(3.5, 30, 'Performance drops\nunderwater', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
print("✓ Saved model_performance.png")

# Print detailed summary
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
print("\nDetection Rates (% of images with detections):\n")
print(f"{'Dataset':<15} {'HOG':>10} {'YOLO':>10} {'DeepLabV3':>10} {'UNet':>10}")
print("-" * 60)

for i, dataset in enumerate(datasets):
    print(f"{dataset:<15}", end='')
    for model in models:
        rate = detection_rates[model][i]
        if model == 'UNet':
            print(f" {rate:>8.0f}%*", end='')
        else:
            print(f" {rate:>9.0f}%", end='')
    print()

print("\n* UNet is untrained (random weights) - results are meaningless\n")

print("KEY FINDINGS:")
print("-" * 80)
print("✓ YOLO: Most robust across all environments (9/10 CityScapes, 3/10 NightTime)")
print("✓ DeepLabV3: Works well in daytime (CityScapes), fails at night")
print("✗ HOG: Only works for standing people in good conditions (0% CityScapes driving)")
print("✗ UNet (untrained): Completely useless - detects random class_8 everywhere")
print()
print("NEXT STEPS:")
print("-" * 80)
print("1. Use YOLO for autonomous driving (proven best performance)")
print("2. Train lightweight UNet/MobileNet on SUIM underwater dataset")
print("3. Deploy trained model on Pynq FPGA for real-time underwater segmentation")
print("4. Combine YOLO + trained segmentation for robust underwater autonomy")
print("="*80)
