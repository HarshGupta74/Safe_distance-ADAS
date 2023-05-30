import cv2
import os
import time
import argparse

from yolo import YoloDetection
from datetime import datetime
CONFIG_FILE = None
model = None

def load_config(config_path):
    global CONFIG_FILE
    CONFIG_FILE = eval(open(config_path).read())

def load_model():
    global model
    model = YoloDetection(CONFIG_FILE["model-parameters"]["model-weights"],
                    CONFIG_FILE["model-parameters"]["model-config"],
                    CONFIG_FILE["model-parameters"]["model-names"],
                    CONFIG_FILE["shape"][0],
                    CONFIG_FILE["shape"][1])
    


# Loop through frames
def start_detection(cap):
    
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M.avi")
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width,height))

    if not cap.isOpened() or not out.isOpened():
        print("Error opening video stream or file")
        exit()

    window_name = "Camera Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.mkdir(folder_name)

    fps_count = 0
    start_time = time.time()
    while True:
        # Read a new frame from the camera
        ret, frame = cap.read()
        # Check if frame was successfully read
        if not ret:
            print("Error reading frame")
            break
        if(ret):
            detections = model.process_frame(frame)
            for output in detections:
                class_id = output[0]
                (x,y,w,h) = output[1:5]
                texx=w*h
                d=(833*1695/w)/1000  #honda_amaze_width =1695mm    d=real_object_width*focal_len/pixel_width 
                #d=30*73/w
                d=round(d)
                centx=str((2*x+w)/2)
                centy=str((2*y+h)/2)
                cent=centx+','+centy          
                cv2.rectangle(frame,(x,y),(x+w,y+h),thickness=2,color=(255,0,0))
                cv2.putText(frame,str(class_id),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0),1)
                cv2.putText(frame,str(texx),(x+w-20,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1)
                cv2.putText(frame,str(d),(x,y+40),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,0,0),1)
                cv2.putText(frame,str(cent),(x,y+20),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1)
                if d<=7:
                    cv2.putText(frame,"WARNING",(int(x+w/2),int(y+h)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
                                
            fps_count += 1
            elapsed_time = time.time() - start_time
            fps = round(fps_count / elapsed_time)
            fps_text = f"FPS: {fps}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    
            cv2.imshow(window_name, frame)

    
            frame_filename = f"{folder_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg"
            cv2.imwrite(frame_filename, frame)

    
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Provide arguements")
    parser.add_argument("--config","-c")
    parser.add_argument("--debug","-d")
    args = parser.parse_args()
    config_path = args.config
    load_config(config_path)
    load_model()
    #fi=time.time()
    pt='test_videos/qw.mp4'
    cap = cv2.VideoCapture(pt)
    #cap =cv2.VideoCapture(0)
    start_detection(cap)
