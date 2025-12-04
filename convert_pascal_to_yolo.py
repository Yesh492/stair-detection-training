#!/usr/bin/env python3
"""
Pascal VOC XML to YOLO Format Converter
CS663 Mobile Vision Project - StairVision

Converts Pascal VOC XML annotations to YOLO format for training.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_voc_to_yolo(xml_file, output_file, class_mapping={'stairs': 0}):
    """
    Convert Pascal VOC XML annotation to YOLO format.
    
    Args:
        xml_file: Path to XML annotation file
        output_file: Path to output YOLO txt file
        class_mapping: Dictionary mapping class names to indices
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_mapping:
            continue
            
        class_id = class_mapping[class_name]
        
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Convert to YOLO format (normalized center x, y, width, height)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))

def batch_convert(xml_dir, output_dir):
    """
    Batch convert all XML files in directory.
    
    Args:
        xml_dir: Directory containing XML files
        output_dir: Directory to save YOLO txt files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = Path(xml_dir).glob('*.xml')
    for xml_file in xml_files:
        output_file = Path(output_dir) / f"{xml_file.stem}.txt"
        convert_voc_to_yolo(str(xml_file), str(output_file))
        print(f"Converted: {xml_file.name} -> {output_file.name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python convert_pascal_to_yolo.py <xml_dir> <output_dir>")
        sys.exit(1)
    
    xml_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    batch_convert(xml_dir, output_dir)
    print(f"\nConversion complete! Check {output_dir} for YOLO format files.")
