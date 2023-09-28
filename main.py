from utils import get_classes, detect_video
from model.yolo_model import YOLO

def main():
    yolo = YOLO(0.3, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)

    detect_video("input.mp4", yolo, all_classes)

if __name__ == "__main__":
    main()
