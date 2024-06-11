from ultralytics import YOLO
import numpy as np
import serial
import torch
import math
import cv2
import pyfirmata2 as pf

class Detection:
    def __init__(self, capture_index):
        # Initialize the webcam and get a reference to the video stream
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.ballModel = self.load_ballModel()
        self.siloModel = self.load_siloModel()


    def load_ballModel(self):
        #  Load ballModel
        ballModel = YOLO("G:\Shared drives\KRAI 2024\Progress\Auwiwir\RedBall2.pt")
        ballModel.fuse()
        return ballModel
    

    def load_siloModel(self):
        ncnn_model = YOLO("G:\Shared drives\KRAI 2024\Progress\Auwiwir\silo_3_class.pt")
        ncnn_model.fuse()
        return ncnn_model


    def predict(self, frame):
        # Implementing ballModel to the frame
        ballResults = self.ballModel(frame, conf = 0.25)
        siloResults = self.siloModel(frame, conf = 0.25)

        # print(f"inference ball: {ballResults}")
        print(f"inference silo: {siloResults}")
        return ballResults, siloResults


    def plot_ball_bboxes(self, results, frame):
        
        xyxys = []
        closeBid = []
        nonZero = 0
        length = 640
        areaBox = 0
        
        # Importing the color masking
        # colorFrame, b, g, r = self.color_masking(frame)
        centerArea = self.center_frame_parameter(results, frame)
        centerArea
        
        for result in results:
            
            # Accessing the bounding boxes  of objects
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy

            # print(result)
            print(xyxys)
    
            for x, y, x1, y1 in xyxys:

                # loop through every pixel of the bounding boxes
                # for width in range(int(x), int(x1)):
                #     for height in range(int(y), int(y1)):
                #         # count the non-zero pixel
                #         if cv2.countNonZero(colorFrame[height, width]) > 0:
                #             nonZero += 1
                #         else:
                #             nonZero = nonZero
                
                areaBoxes = (y1-y)*(x1-x)


                # condition to only draw a bbox to valid object
                # if nonZero >= (areaBoxes/6):

                    # Call a function for the center and length bbox
                coordinate = [x, y, x1, y1]
                centerBox, lengths = self.center_bbox(coordinate, frame)

                # if lengths < 0:
                #     lengths = lengths*(-1)

                print(length)
                print(lengths)

                if abs(areaBox) < abs(areaBoxes):
                    areaBox = areaBoxes
                    length = lengths
                    closeBid = [i for i in coordinate]
                    
                # else:
                #     cv2.rectangle(frame, (int(x), int(y)), (int(x1), int(y1)), (0, 0, 0), 2)

                nonZero = 0

        if length != 90 and len(closeBid) != 0:
            print(closeBid)
            cv2.rectangle(frame, (int(closeBid[0]), int(closeBid[1])), (int(closeBid[2]), int(closeBid[3])), (0, 255, 0), 2)

            # Send objects length through serial here

        print(f"Nilai non-zero: {nonZero}")
            
        return frame, length
    

    def plot_silo_bboxes(self, results, frame):
    
        closeBid = []
        length = 640

        for result in results:

            boxes = result.boxes.cpu().numpy()
            class_ids = boxes.cls
            xyxys = boxes.xyxy

            for i, (x, y, x1, y1) in enumerate(xyxys):

                coordinate = [x, y, x1, y1]
                centerBox, lengths = self.center_bbox(coordinate, frame)
                id_name = result.names[int(class_ids[i])]

                print(length)
                print(lengths)
                
                if id_name == "Silo-1":
                    length = lengths
                    closeBid = [i for i in coordinate]
                    class_id = id_name                        
                    # print(f"class id: {class_ids[i]}")
                elif id_name == "Silo-0":
                    length = lengths
                    class_id = id_name
                    closeBid = [i for i in coordinate]
                        
        if length != 90 and len(closeBid) != 0:
            print(closeBid)
            cv2.rectangle(frame, (int(closeBid[0]), int(closeBid[1])), (int(closeBid[2]), int(closeBid[3])), (50, 205, 50), 2)
            cv2.putText(frame, str(class_id), (int(closeBid[0]), int(closeBid[3]+15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 57), 2, cv2.LINE_AA, False)

        return frame, length



    def color_masking(self, frame):
        # Blue Color
        # light_color = np.array([90, 100, 100])
        # dark_color = np.array([115, 255, 255])

        # #Red Color
        # light_color = np.array([165, 180, 180])
        # dark_color = np.array([180, 255, 255])

        #Red Color (testing)
        light_color = np.array([0, 25, 75])
        dark_color = np.array([8, 255, 255])

        # Map HSV values to RGB light to dark color range
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, light_color, dark_color)
        b, g, r = [0, 255, 0]

        # Masking the video stream with the color mask
        color_result = cv2.bitwise_and(frame, frame, mask=color_mask)

        # Return the mask result and the bgr value for bbox
        return color_result, b, g, r


    def center_bbox(self, coordinate, frame):
        # Extract the coordinate of a valid object
        x, y, x1, y1 = coordinate

        cx = int((x + x1)/2)   # Center point of X coordinates
        cy  = int((y + y1)/2)   # Center point of Y coordinates

        center = [cx, cy]
        centerFrame = [320, 480]

        # Call a function to calculate the length of a valid object
        errorParam = self.object_length(center, centerFrame)
        error = errorParam[0]
        
        # Create the bbox and line to the valid object
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        cv2.line(frame, (cx, cy), (320, cy), (0, 0, 0), 2)
        cv2.putText(frame, str(round(errorParam[0], 2)), (int(x+10), int(y+20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1/2, (37, 95, 255), 2, cv2.LINE_AA, False)
        
        return frame, error
    

    def object_length(self, center, centerFrame):
        # radians = 180
        # phi = 3.14

        # Extracting the coordinate of center of object and frame 
        cx, cy = center
        fx, fy = centerFrame

        # Calculating every side of the triangle that is formed by object center and frame
        horizontal = cx-fx
        vertikal = cy-fy

        # lengthObject = math.sqrt(pow(vertikal, 2) + pow(horizontal, 2))
        # Calculating the length that  the triangle makes with the horizontal axis
        # lengthObject = int((math.acos(samping/miring)*radians)/phi)

        # Condition for when the object on the right side of the center frame
        # if cx > fx:
        #     horizontal *= -1

        return [horizontal, vertikal]


    def center_frame_parameter(self, results, frame):
        # Define the top left and bottom right area 
        halfFrame = int((640/2))

        cv2.line(frame, (halfFrame, 0), (halfFrame, 480) ,(200, 0, 200), 2)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        
        assert cap.isOpened()
        
        a = 0

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            # Using the Model
            predBallFrame, predSiloFrame = self.predict(frame)


            # Integrating bounding boxes and color detection to the frame
            ballFrame, errBall = self.plot_ball_bboxes(predBallFrame, frame)
            siloFrame, errSilo = self.plot_silo_bboxes(predSiloFrame, frame)
            color_frame, b, g, r = self.color_masking(frame)
            
            print(f"Error Bola: {errBall}")
            print(f"Error Silo: {errSilo}")
            if(a % 2 == 0):
                serialInst.write('I'.encode())
                serialInst.write(f'{errBall}'.encode())
            else:
                serialInst.write('I'.encode())
                serialInst.write(f'{errSilo}'.encode())     
            try:
                response = serialInst.readline().decode('utf-8', errors='ignore').strip() # Read the response from Arduino
                print(f"Response Received: {response}")
            except UnicodeDecodeError as e:
                print("UnicodeDecodeError:", e)

            print(f'a = {a}')

            # Display
            cv2.imshow('Ball Detection', ballFrame)
            cv2.imshow("Silo Frame", siloFrame)
            cv2.imshow('Color Frame', color_frame)
            
            # Press q to close the video
            if cv2.waitKey(1) == ord('q'):
                break

            a += 1

        cap.release()
        cv2.destroyAllWindows()

# # Setting serial communication
portVar = "COM12"
serialInst = serial.Serial(port=portVar, baudrate=115200, timeout=0.1)

# 0 capture index for builtin camera, 1 capture index for webcam
yolo_detector = Detection(capture_index=0)
yolo_detector()