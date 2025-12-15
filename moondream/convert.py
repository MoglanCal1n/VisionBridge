import os
import json

BASE_DATA_DIR = "."
QUESTION = "What are the bounding boxes for the sidewalk?"

def convert_yolo_to_json(data_split, output_file):
    """
    converts data from yolo txt into an json file for the VQA
    """

    data_dir = os.path.join(BASE_DATA_DIR, data_split)
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    with open(output_file, 'w', encoding="utf-8") as fout:
        count = 0

        for img_name in os.listdir(images_dir):
            if not img_name.endswith( ('.jpg', '.png', '.jpeg') ):
                continue

            image_path = os.path.join(images_dir, img_name) 

            label_name = os.path.splitext(img_name)[0]+ '.txt'
            label_path = os.path.join(labels_dir, label_name)

            if os.path.exists(label_path):
                with open(label_path, 'r') as fin:
                    line = fin.readline().strip()

                    if not line:
                        continue 

                    parts = line.split()

                    answer_string = f"[{parts[1]}, {parts[2]}, {parts[3]}, {parts[4]}]"

                    json_line= {
                        "image_path": image_path,
                        "question": QUESTION,
                        "answer": answer_string
                    }

                    fout.write(json.dumps(json_line) + '\n')
                    count += 1

if __name__ == "__main__":
    convert_yolo_to_json('train', 'train.jsonl')
    convert_yolo_to_json('valid', 'valid.jsonl')
    convert_yolo_to_json('test', 'test.jsonl')

    print("Gata")
