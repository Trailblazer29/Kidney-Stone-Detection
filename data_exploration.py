from glob import glob
import cv2, random

if __name__=="__main__":

    path_to_images = "data/train/images"
    path_to_labels = "data/train/labels"

    img_paths = glob(path_to_images + "/*.jpg")
    label_paths = glob(path_to_labels + "/*.txt")

    # Display random image
    idx = random.randint(0,len(img_paths)-1)
    img_path = img_paths[idx]
    label_path = label_paths[idx]

    image = cv2.imread(img_path)
    image_height, image_width, _ = image.shape

    with open(label_path, "r") as label_file:
        lines = label_file.readlines()
        for line in lines:
           center, x_center, y_center, box_width, box_height = line.split(" ")
           # print(x_center, y_center, box_width, box_height)

           # Denormalize image coordinates and annotation box dimensions
           x_center, y_center, box_width, box_height = float(x_center), float(y_center), float(box_width), float(box_height)
        
           x_center *= image_width
           y_center *= image_height
           box_width *= image_width
           box_height *= image_height
           # print(x_center, y_center, box_width, box_height)

           # Draw annotation boxes on kidney stones
           p1_x = int(x_center - box_width/2)
           p1_y = int(y_center - box_height/2)
           p2_x = int(x_center + box_width/2)
           p2_y = int(y_center + box_height/2)

           cv2.rectangle(image, [p1_x, p1_y], [p2_x, p2_y], (0, 255, 0), 1)
    
    cv2.imshow("Kidney X-Ray", image)
    cv2.waitKey(0)
