from ultralytics import YOLO
import cv2

class KidneyStoneDetectionModel:

    def __init__(self, model_path) -> None:        
        self.model = YOLO(model=model_path)
        self.results = []

    def run_inference(self, image):
        self.results = self.model(image, conf=0.5)

    def annotate_image(self, image):
        annotated_image = image.copy()
        for result in self.results:
            boxes = result.boxes
            for box in boxes:
                coord = box.xyxy[0]
                x1, y1 = int(coord[0]), int(coord[1])
                x2, y2 = int(coord[2]), int(coord[3])
                cv2.rectangle(annotated_image, [x1, y1], [x2, y2], (0, 255, 0), 1)
        return annotated_image
    
if __name__=="__main__":
    model_path = "./ks_detection.pt"

    print("Loading model..")
    model = KidneyStoneDetectionModel(model_path=model_path)

    img_path = "./sample_image.jpg"

    print("Reading image..")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Running inference..")
    model.run_inference(image=image)

    print("Annotating image..")
    annotated_image = model.annotate_image(image=image)

    cv2.imshow("results", annotated_image)
    cv2.waitKey(0)