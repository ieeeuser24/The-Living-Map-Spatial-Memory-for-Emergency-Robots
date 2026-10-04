import os, json, shutil

SRC = r"C:\Users\nour limem\Downloads\victim_clean"
DST = r"C:\Users\nour limem\Downloads\victim_upload"
N = 1000   # max images per upload folder

for split in ["train", "valid", "test"]:
    src = os.path.join(SRC, split)
    with open(os.path.join(src, "bounding_boxes.labels")) as f:
        boxes = json.load(f)["boundingBoxes"]
    names = sorted(boxes)
    for b in range(0, len(names), N):
        chunk = names[b:b + N]
        out = os.path.join(DST, f"{split}_{b // N + 1:02d}")
        os.makedirs(out, exist_ok=True)
        new_boxes = {}
        for k, old in enumerate(chunk, start=b):
            new = f"{split}_{k:05d}.jpg"
            shutil.copy(os.path.join(src, old), os.path.join(out, new))
            new_boxes[new] = boxes[old]
        with open(os.path.join(out, "bounding_boxes.labels"), "w") as f:
            json.dump({"version": 1, "type": "bounding-box-labels",
                       "boundingBoxes": new_boxes}, f)
        print(out, len(chunk))