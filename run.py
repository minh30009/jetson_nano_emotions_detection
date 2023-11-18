import mediapipe as mp
import time
import cv2
import numpy as np
 

import tensorflow as tf

idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
interpreter = tf.lite.Interpreter(model_path="./emotions_mobilenet.tflite")
interpreter.allocate_tensors()
    # Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
 
        self.minDetectionCon = minDetectionCon
 
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
 
    def findFaces(self, img, draw=True):
 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img, max = self.fancyDraw(img,bbox)
 
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
                    cv2.putText(img, str(idx_to_class[max]), (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return img, bboxs
 
    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

 
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        crop_image = img[y:y1, x:x1]
        img_resized = cv2.resize(src=crop_image, dsize=(224, 224))
        input_image = img_resized.astype(np.float32)
        input_image = np.expand_dims(input_image, 0)
        interpreter.set_tensor(input_details[0]['index'], input_image)  
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index']) 
        max = np.argmax(output_data)

        return img, max
 
 
def main():
    #video_capture = cv2.VideoCapture(0)
    


    #cap = cv2.VideoCapture("Videos/6.mp4")
    #ret, cap = video_capture.read()
    #vs = VideoStream(src=0).start()
    video_capture = cv2.VideoCapture(0)
    time.sleep(2.0)
    frame = video_capture.read()
    #frame = imutils.resize(frame, width=500)
    pTime = 0
    detector = FaceDetector()
    while True:
        #success, img = cap.read()
        #ret, img = video_capture.read()
        #frame = vs.read()
        
        ret, frame = video_capture.read()

        img, bboxs = detector.findFaces(frame)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 
 
if __name__ == "__main__":
    main()
