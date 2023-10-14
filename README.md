# GraduationThesis
This my Graduation Thesis Project. This projects aims to detect circular objects at images. To train and predict I used YOLOv8 models. Reason of the selecting YOLO model is YOLO has a robotust performance on Image Detection.

I created the dataset own my own.
  * Firstly, I searched circular objects from google then downloaded them
  * After that I used the Roboflow to implementing the segmentation of circular object(s) on image and do some image editing(like reshaping, etc.)
  * Finally I downloaded the final dataset to train the YOLOv8 models.

User can select among the YOLOv8 models via usimg the basic GUI. Then select epoch numbers to train the. After the training user can insert own image to test trained model.
