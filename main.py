import cv2
import mediapipe as mp
import os

#! Function to process an image or frame
def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    #* _ THIS IS DONE TO GET ONLY HEIGHT AND WIDTH AND WE DON'T NEED THE CHANNELS
    H, W, _ = img.shape

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            
            #* TO SCALE THE BOUNDING BOXES WTR TO  THE IMAGE SIZE
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            
            img[y1:y1+h, x1:x1+w] = cv2.blur(img[y1:y1+h, x1:x1+w], (50, 50))
            
    return img




#! Function to process an image file
def process_image(face_detection, output_folder):
    img_path = input("Please paste the full path to the image file: ").strip()
    img = cv2.imread(img_path)
    if img is None:
        print("Invalid image path! Please make sure the file exists.")
        return
    processed_img = process_img(img, face_detection)
    output_path = os.path.join(output_folder, "blurred_image.jpg")
    cv2.imwrite(output_path, processed_img)
    print(f"Blurred image saved at {output_path}")




# Function to process a video file
def process_video(face_detection, output_folder):
    video_path = input("Please paste the full path to the video file: ").strip()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Invalid video path! Please make sure the file exists.")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #* FOURCC IS A 4 BYTE CODE USED TO SPECIFY VIDEO CODEC
    output_path = os.path.join(output_folder, "blurred_video.avi")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_img(frame, face_detection)
        out.write(processed_frame)
    
    cap.release()
    out.release()
    print(f"Blurred video saved at {output_path}")





#! Function to process webcam feed
def process_webcam(face_detection, output_folder):
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Default FPS for webcam feed
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(output_folder, "blurred_webcam.avi")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print("Recording webcam feed. Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No frame captured. Continuing...")
            continue
        
        processed_frame = process_img(frame, face_detection)
        out.write(processed_frame)
        cv2.imshow("Webcam - Blurred Faces", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Blurred webcam video saved at {output_path}")




#! Main function to handle user choice
def main():
    output_folder = os.path.expanduser("~/Downloads")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print("Choose an option:")
    print("1. Process an image")
    print("2. Process a video")
    print("3. Use the webcam")
    choice = int(input("Enter your choice (1/2/3): "))

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if choice == 1:
            process_image(face_detection, output_folder)
        elif choice == 2:
            process_video(face_detection, output_folder)
        elif choice == 3:
            process_webcam(face_detection, output_folder)
        else:
            print("Invalid choice! Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
