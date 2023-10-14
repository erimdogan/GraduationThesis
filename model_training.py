from ultralytics import YOLO
import multiprocessing as mp

if __name__ == '__main__':
    mp.freeze_support()
    # Loading model 
    model = YOLO('yolov8s.yaml')
    model = YOLO('yolov8s.pt')
    model = YOLO('yolov8s.yaml').load('yolov8s.pt')
    #train model
    model.train(data='data.yaml', epochs=100, imgsz=640)