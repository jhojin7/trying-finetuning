#!/usr/bin/env python3
"""Quick YOLO training on synthetic dataset"""
import json
from pathlib import Path
from ultralytics import YOLO

# Convert COCO to YOLO format quickly
def convert_coco_to_yolo(coco_json, img_dir, output_dir):
    """Convert COCO annotations to YOLO format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json) as f:
        data = json.load(f)

    # Create image_id to annotations mapping
    img_dict = {img['id']: img for img in data['images']}

    for ann in data['annotations']:
        img_id = ann['image_id']
        img_info = img_dict[img_id]

        # COCO bbox: [x, y, width, height]
        x, y, w, h = ann['bbox']
        img_w, img_h = img_info['width'], img_info['height']

        # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h

        # Write to file
        label_file = output_dir / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_file, 'a') as f:
            f.write(f"{ann['category_id']} {x_center} {y_center} {width} {height}\n")

    print(f"✅ Converted {len(data['images'])} images")

print("🔄 Converting dataset to YOLO format...")
dataset_dir = Path("dataset")
yolo_dir = Path("dataset_yolo")

for split in ['train', 'valid', 'test']:
    convert_coco_to_yolo(
        dataset_dir / split / "_annotations.coco.json",
        dataset_dir / split,
        yolo_dir / "labels" / split
    )
    # Copy images
    import shutil
    (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    for img in (dataset_dir / split).glob("*.jpg"):
        shutil.copy(img, yolo_dir / "images" / split / img.name)

# Create data.yaml
yaml_content = f"""
path: {yolo_dir.absolute()}
train: images/train
val: images/valid
test: images/test

nc: 5
names: ['red', 'blue', 'green', 'yellow', 'orange']
"""

(yolo_dir / "data.yaml").write_text(yaml_content)
print(f"✅ Dataset ready at {yolo_dir.absolute()}")

# Train
print("\n🚀 Starting YOLO training (5 epochs, fast)...")
model = YOLO('notebooks/yolov8n.pt')
results = model.train(
    data=str(yolo_dir / "data.yaml"),
    epochs=5,
    imgsz=640,
    batch=8,
    device='cpu',
    project='outputs',
    name='yolo_synthetic',
    exist_ok=True
)

print("\n✅ Training complete!")
print(f"📁 Results: outputs/yolo_synthetic")

# Quick test
print("\n🧪 Testing inference...")
test_img = list((dataset_dir / "test").glob("*.jpg"))[0]
results = model(test_img)
results[0].save(f"outputs/yolo_synthetic/test_prediction.jpg")
print(f"✅ Test prediction saved: outputs/yolo_synthetic/test_prediction.jpg")
