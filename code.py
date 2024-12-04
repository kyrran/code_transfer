# Custom PASCAL VOC Dataset for YOLO
path: ../datasets/VOC  # Root dataset path
train: ImageSets/Main/train.txt  # Relative path to training split
val: ImageSets/Main/val.txt      # Relative path to validation split
test: ImageSets/Main/test.txt    # Relative path to testing split (optional)

# Classes
names:
  0: jg       # Manhole Cover (井盖)
  1: rxd      # Crossing Light (人行灯)
  2: dxgx     # Pipeline Indicating Pile (地下管线桩)
  3: zsp      # Traffic Signs (指示牌)
  4: xfs      # Hydrant (消防栓)
  5: dzy      # Camera (电子眼)
  6: lhd      # Traffic Light (红绿灯)
  7: jdp      # Guidepost (街道路名牌)
  8: jsp      # Traffic Warning Sign (警示牌)
  9: ld       # Streetlamp (路灯)
  10: txx     # Communication Box (通讯箱)

# Optional download script for automatic setup ---------------------------------------------------------------------------------------
download: |
  import os
  import xml.etree.ElementTree as ET
  from tqdm import tqdm
  from pathlib import Path

  def convert_voc_to_yolo(annotation_path, labels_path, class_mapping):
      """Convert VOC XML annotations to YOLO format."""
      tree = ET.parse(annotation_path)
      root = tree.getroot()

      # Image size
      size = root.find("size")
      w = int(size.find("width").text)
      h = int(size.find("height").text)

      with open(labels_path, "w") as label_file:
          for obj in root.findall("object"):
              cls_name = obj.find("name").text
              if cls_name not in class_mapping:
                  continue

              cls_id = class_mapping[cls_name]
              bbox = obj.find("bndbox")
              xmin = float(bbox.find("xmin").text)
              xmax = float(bbox.find("xmax").text)
              ymin = float(bbox.find("ymin").text)
              ymax = float(bbox.find("ymax").text)

              # YOLO format: class_id x_center y_center width height (normalized)
              x_center = ((xmin + xmax) / 2) / w
              y_center = ((ymin + ymax) / 2) / h
              box_width = (xmax - xmin) / w
              box_height = (ymax - ymin) / h

              label_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

  # Paths
  dataset_root = Path(yaml['path'])
  annotations_dir = dataset_root / "Annotations"
  images_dir = dataset_root / "JPEGImages"
  labels_dir = dataset_root / "labels"
  labels_dir.mkdir(exist_ok=True, parents=True)

  # Class mapping
  class_mapping = {name: idx for idx, name in yaml['names'].items()}

  # Convert annotations
  for annotation_file in tqdm(os.listdir(annotations_dir), desc="Converting VOC to YOLO"):
      if annotation_file.endswith(".xml"):
          annotation_path = annotations_dir / annotation_file
          label_path = labels_dir / f"{os.path.splitext(annotation_file)[0]}.txt"
          convert_voc_to_yolo(annotation_path, label_path, class_mapping)
