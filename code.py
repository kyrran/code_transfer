import os
import xml.etree.ElementTree as ET
import shutil

# Configuration
VOC_ROOT = "./VOC"  # Root directory of your VOC dataset
OUTPUT_DIR = "./YOLO_Kaggle"  # Output directory for YOLO-formatted dataset
SPLITS_DIR = os.path.join(VOC_ROOT, "ImageSets/Main")  # Directory containing train.txt, val.txt, test.txt
CLASSES = {
    "jg": 0,       # Manhole Cover (井盖)
    "rxd": 1,      # Crossing Light (人行灯)
    "dxgx": 2,     # Pipeline Indicating Pile (地下管线桩)
    "zsp": 3,      # Traffic Signs (指示牌)
    "xfs": 4,      # Hydrant (消防栓)
    "dzy": 5,      # Camera (电子眼)
    "lhd": 6,      # Traffic Light (红绿灯)
    "jdp": 7,      # Guidepost (街道路名牌)
    "jsp": 8,      # Traffic Warning Sign (警示牌)
    "ld": 9,       # Streetlamp (路灯)
    "txx": 10      # Communication Box (通讯箱)
}

def voc_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0 - 1
    y_center = (box[2] + box[3]) / 2.0 - 1
    width = box[1] - box[0]
    height = box[3] - box[2]
    return x_center * dw, y_center * dh, width * dw, height * dh

def convert_annotation(xml_path, output_txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    with open(output_txt_path, "w") as txt_file:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in CLASSES:
                print(f"Skipping unrecognized class: {class_name}")
                continue
            class_id = CLASSES[class_name]
            bndbox = obj.find("bndbox")
            box = (
                float(bndbox.find("xmin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("ymax").text)
            )
            yolo_box = voc_to_yolo((img_width, img_height), box)
            txt_file.write(f"{class_id} {' '.join(f'{coord:.6f}' for coord in yolo_box)}\n")

def process_split(split_name, split_file, output_images_dir, output_labels_dir, annotations_dir, images_dir):
    with open(split_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines()]

    split_image_dir = os.path.join(output_images_dir, split_name)
    split_label_dir = os.path.join(output_labels_dir, split_name)
    os.makedirs(split_image_dir, exist_ok=True)
    os.makedirs(split_label_dir, exist_ok=True)

    for image_id in image_ids:
        xml_path = os.path.join(annotations_dir, f"{image_id}.xml")
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        output_image_path = os.path.join(split_image_dir, f"{image_id}.jpg")
        output_label_path = os.path.join(split_label_dir, f"{image_id}.txt")

        # Convert annotation
        if os.path.exists(xml_path):
            convert_annotation(xml_path, output_label_path)
        else:
            print(f"Warning: Annotation file not found for {image_id}")

        # Copy image to YOLO folder
        if os.path.exists(image_path):
            shutil.copy(image_path, output_image_path)
        else:
            print(f"Warning: Image file not found for {image_id}")

def main():
    annotations_dir = os.path.join(VOC_ROOT, "Annotations")
    images_dir = os.path.join(VOC_ROOT, "JPEGImages")
    output_images_dir = os.path.join(OUTPUT_DIR, "images")
    output_labels_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Process each split
    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(SPLITS_DIR, f"{split_name}.txt")
        if os.path.exists(split_file):
            print(f"Processing {split_name} split...")
            process_split(split_name, split_file, output_images_dir, output_labels_dir, annotations_dir, images_dir)
        else:
            print(f"Skipping {split_name} split (file not found)")

    print(f"YOLO dataset created at {OUTPUT_DIR}")
    print("Images and annotations are ready for training.")

if __name__ == "__main__":
    main()
