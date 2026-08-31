import os
import xml.etree.ElementTree as ET

xml_folder = r"C:\Users\asifm\road_damage_detection\Dataset\images\val"
output_folder = r"C:\Users\asifm\road_damage_detection\Dataset\labels\val"

print("XML folder:", xml_folder)

if not os.path.exists(xml_folder):
    print(" XML folder not found")
    exit()

os.makedirs(output_folder, exist_ok=True)

class_mapping = {
    "D10": 0,
    "D00": 1,
    "D20": 2,
    "D40": 3
}

files = os.listdir(xml_folder)
print("Total XML files found:", len(files))

for file in files:
    if not file.endswith(".xml"):
        continue

    print("Processing:", file)

    tree = ET.parse(os.path.join(xml_folder, file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    with open(os.path.join(output_folder, file.replace(".xml", ".txt")), "w") as f:
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in class_mapping:
                continue

            cid = class_mapping[label]
            box = obj.find("bndbox")

            xmin = int(box.find("xmin").text)
            xmax = int(box.find("xmax").text)
            ymin = int(box.find("ymin").text)
            ymax = int(box.find("ymax").text)

            xc = ((xmin + xmax) / 2) / w
            yc = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            f.write(f"{cid} {xc} {yc} {bw} {bh}\n")

print(" Conversion finished successfully")