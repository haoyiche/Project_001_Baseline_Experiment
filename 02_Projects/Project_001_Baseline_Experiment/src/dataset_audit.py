import os
from PIL import Image


def scan_dataset(root):

    image_count = 0
    sizes = []
    channels = {}
    classes = {}


    for dirpath,_,filenames in os.walk(root):

        for file in filenames:

            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

                image_path = os.path.join(dirpath,file)
                try:
                    image = Image.open(image_path)
                    image_count += 1
                    sizes.append(image.size)
                    mode = image.mode
                    channels[mode] = channels.get(mode, 0) + 1
                    class_name = os.path.basename(dirpath)
                    classes[class_name] = classes.get(class_name, 0) + 1


                except Exception as e:
                    print(f"Error opening image {image_path}: {e}")
    return (image_count, sizes, channels, classes)

def save_report(count, sizes, channels, classes, output_file="./results/dataset_report.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("====================\n")
        f.write("Dataset Audit Report\n")
        f.write("====================\n")
        f.write(f"Total images: {count}\n")
        f.write(f"Image sizes: {sizes[:5]}\n")
        f.write(f"Channel distribution: {channels}\n")
        for k,v in channels.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"Class distribution: {classes}\n")
        for k,v in classes.items():
            f.write(f"  {k}: {v}\n")

if __name__ == "__main__":
    root = "data/raw"

    count, sizes, channels, classes  = scan_dataset(root)

    print("====================")
    print("Dataset Audit")
    print("====================")
    print(f"Total images: {count}")
    print(f"Image sizes: {sizes[:5]}")
    print(f"Channel distribution: {channels}")
    print(f"Class distribution: {classes}")

    save_report(count, sizes, channels, classes, output_file="./results/dataset_report.txt")
    print("Report saved to ./results/dataset_report.txt")
