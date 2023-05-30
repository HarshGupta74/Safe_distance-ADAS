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
    # Set up video writer
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M.avi")
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width,height))
     # Check if video stream and writer are opened
    if not cap.isOpened()or not out.isOpened():
        print("Error opening video stream or file")
        exit()

    # Set up window for preview
    window_name = "C Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create a new folder for saving frames
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.mkdir(folder_name)

    # Initialize FPS counter
    fps_count = 0
    start_time = time.time()

    # Loop through frames
    while True:
        # Read a new frame from the camera
        # t1=time.time()
        ret, frame = cap.read()

        # Check if frame was successfully read
        if not ret:
            print("End of video")
            break
        
        # Process frame using YOLO model
        detections = model.process_frame(frame)

        # Loop through all detections in the frame
        for output in detections:
            # Extract class ID, bounding box coordinates, and object's area
            class_id = output[0]
            x, y, w, h = output[1:5]
            area = w * h

            # Calculate centroid coordinates
            cent_x, cent_y = x + w / 2, y + h / 2            
            # Draw bounding box, class label, area, and centroid on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), thickness=2, color=(255,0,0))
            cv2.putText(frame, str(class_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
            cv2.putText(frame, f"Width: {w:.0f}", (x, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
            cv2.putText(frame, f"Height: {h:.0f}", (x, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
            cv2.putText(frame, f"Centroid: ({cent_x:.0f}, {cent_y:.0f})", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)

        # Calculate FPS and draw it on the frame
        fps_count += 1
        elapsed_time = time.time() - start_time
        fps = round(fps_count / elapsed_time)
        fps_text = f"FPS: {fps}"
        # print(fps)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, fps_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame in a window
        cv2.imshow(window_name, frame)

        # Save the frame to the folder
        frame_filename = f"{folder_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg"
        cv2.imwrite(frame_filename, frame)

        # Write the frame to the video file
        out.write(frame)
        # t2=time.time()
        # print("time each: " ,t2-t1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # end_time = time.time()
    # total_time = end_time - start_time
    # print("Total time taken: ", total_time)
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
    cap = cv2.VideoCapture('test_videos/game.mp4')
    start_detection(cap)
