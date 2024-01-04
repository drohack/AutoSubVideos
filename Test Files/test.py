import time
import cv2
import easyocr
import tqdm

print("[INFO] starting video file thread...")
cap = cv2.VideoCapture("C:\\Users\\droha\\Videos\\One Piece - S01E01 - Intro.m4v")

reader = easyocr.Reader(['ja'], gpu=True, quantize=True)  # Specify language(s)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm.tqdm(total=total_frames, desc="Processing Frames", unit="frame")

start_time = time.time()

frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if frame_counter % 10 == 0:  # process every 10th frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #results = reader.readtext(gray_frame, paragraph=True, batch_size=5)
            detected = reader.detect(gray_frame)
            #print(detected)
        progress_bar.update(1)
        frame_counter += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] elasped time: {:.2f}ms".format(time.time() - start_time))