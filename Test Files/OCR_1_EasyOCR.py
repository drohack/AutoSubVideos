import os
import tkinter
import datetime
from tkinter import filedialog

import easyocr
import cv2
import numpy
from pylab import rcParams
import jellyfish as jf


RESULT_CONFIDENCE = 0.0
JELLYFISH_ACCURACY = 0.7


def print_with_timestamp(message):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(formatted_time, message)


def ocr_video(video_file_path):
    print_with_timestamp("Start ocr_video(" + video_file_path + ")")

    frames = []
    reader_output = []

    # Setup EasyOCR Reader
    reader = easyocr.Reader(['ja'])  # Specify language(s)
    # Load video file to cv2
    cap = cv2.VideoCapture(video_file_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Loop through each frame from video
    i = 0  # frame number
    while cap.isOpened():
        ret, frame = cap.read()

        # Make sure the frame exists
        if ret:
            if i % 6 == 0:  # process every 6 frames
                # Run EasyOCR on frame
                results = reader.readtext(frame, paragraph=True)

                # Loop through each text output and add it to an array
                frame_result_array = []
                for result in results:
                    box = result[0]
                    text = result[1]
                    # confidence = result[2]
                    confidence = 1
                    if len(text) > 2 and confidence > RESULT_CONFIDENCE:
                        frame_result_array.append(result)

                # Add the Frame and EasyOCR Reader results in step so the indexes are synced
                frames.append(i)
                reader_output.append(frame_result_array)

                if len(frame_result_array) > 0:
                    print(frame_result_array)

                for result in results:
                    text = result[1]
                    box = result[0]

                    # Ensure integer coordinates for rectangle
                    box[0] = tuple(map(int, box[0]))
                    box[2] = tuple(map(int, box[2]))

                    # Draw rectangle and text
                    cv2.rectangle(frame, box[0], box[2], (0, 255, 0), 2)
                    cv2.putText(frame, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.imshow("Text Detection", frame)

            i += 1  # increase the frame number

            # Wait until the "q" key is pressed
            if cv2.waitKey(1) == ord("q"):
                break
        else:
            # Once the video is finished break out of the loop
            break

    # close video and all cv2 instances
    cap.release()
    cv2.destroyAllWindows()




def create_empty_srt_file(file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("")


# Define the path to the video file
tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
video_path = filedialog.askopenfilename()

print_with_timestamp("Creating ocr file for: " + video_path)

try:
    ocr_video(video_path)
finally:
    print_with_timestamp("done")
