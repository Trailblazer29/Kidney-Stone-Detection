from ultralytics import YOLO

if __name__=="__main__":

    num_epochs = 1
    batch_size = 1

    model = YOLO("yolov8s.pt")
    model.train(data="data.yaml", epochs = num_epochs, batch = batch_size)

    eval_metrics = model.val()