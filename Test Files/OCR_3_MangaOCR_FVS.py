import os
import tkinter
import datetime
from tkinter import filedialog
import tqdm
import traceback

import easyocr
import math

import cv2
import numpy as np
from PIL import Image
from manga_ocr import MangaOcr
import jellyfish as jf

from imutils.video import FileVideoStream
import time


RESULT_CONFIDENCE = 0.0
JELLYFISH_ACCURACY = 0.5
BOX_DISTANCE_REQUIREMENT = 15
BOX_SIZE_REQUIREMENT = 45
MAX_FRAME_JUMP = 15


def print_with_timestamp(message):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(formatted_time, message)


def are_boxes_similar(box1, box2):
    # Extracting coordinates and dimensions
    x1, y1, w1, h1 = box1[0][0], box1[0][1], box1[1][0] - box1[0][0], box1[3][1] - box1[0][1]
    x2, y2, w2, h2 = box2[0][0], box2[0][1], box2[1][0] - box2[0][0], box2[3][1] - box2[0][1]

    # Calculate center points
    center1 = ((x1 + x1 + w1) // 2, (y1 + y1 + h1) // 2)
    center2 = ((x2 + x2 + w2) // 2, (y2 + y2 + h2) // 2)

    # Calculate distance between centers
    distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    # Calculate size difference
    size_difference = abs(w1 - w2) + abs(h1 - h2)

    if distance < BOX_DISTANCE_REQUIREMENT and size_difference < BOX_SIZE_REQUIREMENT:
        return True
    else:
        #print_with_timestamp(f"Distance between centers: {distance}")
        #print_with_timestamp(f"Size difference: {size_difference}")
        return False


def frames_to_time(frame_number, fps):
    time_in_seconds = frame_number / fps

    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(time_in_seconds % 60)
    milliseconds = int((time_in_seconds % 1) * 1000)

    time_format = "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, seconds, milliseconds)
    return time_format


def ocr_video(video_file_path):
    print_with_timestamp("Start ocr_video(" + video_file_path + ")")

    # Reader output holding a frame number, text box coordinates, text
    reader_output = []

    # Setup EasyOCR Reader
    reader = easyocr.Reader(['ja'])  # Specify language(s)

    # Setup MangaOcr Reader
    mocr = MangaOcr()

    fvs = FileVideoStream(video_file_path).start()
    #time.sleep(1.0)

    fps = fvs.stream.get(cv2.CAP_PROP_FPS)
    #print(f'fps: {fps}')

    # Initialize a progress bar
    total_frames = int(fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm.tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    # Loop through each frame from video
    i = 0  # frame number
    try:
        while fvs.more():
            frame = fvs.read()

            # Run EasyOCR on frame
            results = reader.readtext(frame, paragraph=True)
            frame_result_array = []
            for result in results:
                text = result[1]
                box = result[0]

                # Only care about boxes that have text that's longer than 2 characters
                if len(text) > 2:
                    # Convert the box coordinates to a NumPy array
                    np_box = np.array(box)

                    # Crop the frame to the specified box
                    cropped_frame = frame[min(np_box[:, 1]):max(np_box[:, 1]), min(np_box[:, 0]):max(np_box[:, 0])]

                    # Verify that the image exists
                    if cropped_frame is not None and cropped_frame.size != 0:
                        # Convert the OpenCV frame to a PIL.Image object
                        pil_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

                        # Run MangaOCR on frame
                        manga_ocr_text = mocr(pil_image)

                        # Replace EasyOCR text with better MangaOCR result
                        # print_with_timestamp(text + " | " + manga_ocr_text)
                        text = manga_ocr_text

                        # Skip printing the box if the text is illegible
                        if text == '．．．':
                            continue

                        # Add updated results to frame_result_array
                        frame_result_array.append([i, box, text])

                    '''
                    # Ensure integer coordinates for rectangle
                    box[0] = tuple(map(int, box[0]))
                    box[2] = tuple(map(int, box[2]))
    
                    # Draw rectangle and text
                    cv2.rectangle(frame, box[0], box[2], (0, 255, 0), 2)
                    cv2.putText(frame, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
            cv2.imshow("Text Detection", frame)
            '''

            if len(frame_result_array) > 0:
                # Add the Frame and EasyOCR Reader results in step so the indexes are synced
                reader_output.append(frame_result_array)
                #print(frame_result_array)

            # Wait until the "q" key is pressed
            if cv2.waitKey(1) == ord("q"):
                break

            i += 1  # increase the frame number

            progress_bar.update(1)  # Update the progress bar
    except Exception as e:
        print_with_timestamp(f"Error: {e}")
        traceback.print_exc()
    finally:
        # close video and all cv2 instances
        fvs.stop()
        cv2.destroyAllWindows()

    # Loop through all of the reader_output to find the start/end frames of each text box
    # Use the frame number to see if we've jumped ahead in the video, thus having a new text box
    # Then use the box location and size to first see if we are using the same text
    # Then use JellyFish to compare text to see how similar they are
    # If it's the same box and text keep going, counting up the frames till you get to the end frame
    final_text_box_array = []
    active_text_array = []
    for frame_result_array in reader_output:
        #print_with_timestamp(f'____frame____: {frame_result_array[0][0]}')

        # If there's no active text, then add the next frame_results as active text setting their starting
        # and ending frame as the existing frame
        # Else compare the frame_resutls to any active text
        if len(active_text_array) == 0:
            for i in range(len(frame_result_array) - 1, -1, -1):
                frame_result = frame_result_array[i]
                # start_frame, end_frame, box, text
                active_text = [frame_result[0], frame_result[0], frame_result[1], frame_result[2]]
                active_text_array.append(active_text)

            #print_with_timestamp(f'initialize active_text_array: {active_text_array}')
            # Go to next frame_result_array
            continue
        else:
            # Compare each active_text against each frame_result
            for i in range(len(active_text_array) - 1, -1, -1):
                active_text = active_text_array[i]
                new_active_end_frame = active_text[1]

                # If we jumped too many frames, move the active_text to final_text_box_array,
                # and skip to the next active_text
                if frame_result_array is not None and len(frame_result_array) > 0 and len(frame_result_array[0]) > 0 and frame_result_array[0][0] - active_text[1] > MAX_FRAME_JUMP:
                    #print_with_timestamp(f'MAX_FRAME_JUMP - {active_text}')
                    final_text_box_array.append(active_text_array.pop(i))
                    continue

                # Loop through all frame_results and compare them to the active_text
                for j in range(len(frame_result_array) - 1, -1, -1):
                    frame_result = frame_result_array[j]
                    # Check if text boxes locations are similar
                    #print_with_timestamp(f'are_boxes_similar: {are_boxes_similar(active_text[2], frame_result[1])}')
                    if are_boxes_similar(active_text[2], frame_result[1]):
                        # Check if the text is similar using JellyFish
                        #print_with_timestamp(f'jaro_similarity: {jf.jaro_similarity(active_text[3], frame_result[2])}')
                        if jf.jaro_similarity(active_text[3], frame_result[2]) > JELLYFISH_ACCURACY:
                            # Update end frame
                            new_active_end_frame = frame_result[0]
                            # Remove frame_result from array so we don't need to look at it again
                            frame_result_array.pop(j)
                            break

                # If we have a new active end frame update the end frame
                # Else this active_text is finished and needs to be popped off and added to the final_text_box_array
                if new_active_end_frame != active_text[1]:
                    active_text[1] = new_active_end_frame
                    active_text_array[i][1] = new_active_end_frame
                    #print_with_timestamp(f'update end_frame: {active_text}')
                else:
                    final_text_box_array.append(active_text_array.pop(i))
                    #print_with_timestamp(f'pop {active_text}')

            # Loop through remaining frame_resutls and add them to the active_text_array
            # (all text boxes that are continuing from last frame have already been popped off)
            #if len(frame_result_array) > 0:
            #    print_with_timestamp(f'add remaining frame_results: {frame_result_array}')
            for j in range(len(frame_result_array) - 1, -1, -1):
                frame_result = frame_result_array[j]
                # start_frame, end_frame, box, text
                active_text = [frame_result[0], frame_result[0], frame_result[1], frame_result[2]]
                active_text_array.append(active_text)

    # Loop through remaining active_texts and add them to final_text_botx_array
    for i in range(len(active_text_array) - 1, -1, -1):
        final_text_box_array.append(active_text_array.pop(i))

    # Loop through final_text_box_array and remove all 0-6 frame text boxes
    for i in range(len(final_text_box_array) - 1, -1, -1):
        if final_text_box_array[i][1] - final_text_box_array[i][0] <= 3:
            final_text_box_array.pop(i)

    # Sort the array based on the first number in each subarray
    final_text_box_array = sorted(final_text_box_array, key=lambda x: x[0])

    # Loop through final_text_box_array and update the frame start/end times to actual time
    for text_box in final_text_box_array:
        text_box[0] = frames_to_time(text_box[0], fps)
        text_box[1] = frames_to_time(text_box[1], fps)

    for text_box in final_text_box_array:
        print(text_box)


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
