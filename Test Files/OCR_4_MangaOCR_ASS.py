import os
import tkinter
import datetime
from tkinter import filedialog
from tqdm import tqdm
import traceback
import concurrent.futures
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re

import easyocr  # https://www.jaided.ai/easyocr/documentation/
import math

import cv2
import numpy as np
from PIL import Image
from manga_ocr import MangaOcr  # https://github.com/kha-white/manga-ocr
import jellyfish as jf
from collections import Counter


RESULT_CONFIDENCE = 0.0
JELLYFISH_ACCURACY = 0.5        # From 0 to 1, how close 2 lines of text have to be to be considered the same string
BOX_DISTANCE_REQUIREMENT = 30   # Maximum center distance of 2 boxes need to be to be considered the same box
BOX_SIZE_REQUIREMENT = 90       # Maximum area difference of 2 boxes need to be to be considered the same box
MAX_FRAME_JUMP_SEC = 3          # Maximum frames we wait till ending a text box (seconds * fps)
MIN_FRAME_LENGTH_SEC = 0.7      # Minimum duration we keep text boxes (we throw out what's lower than this) (seconds * fps)


def get_device() -> str:
    return 'cuda'


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
    milliseconds = int((time_in_seconds % 1) * 100)

    time_format = "{:02d}:{:02d}:{:02d}.{:02d}".format(hours, minutes, seconds, milliseconds)
    return time_format


def translate_text_box(text_box, tokenizer, model, device, pbar) -> str:
    encoded_text = tokenizer(text_box[3], return_tensors="pt")
    # Run on GPU if available
    encoded_text = encoded_text.to(device)

    translated_text = model.generate(**encoded_text)[0]
    translated_sentence = tokenizer.decode(translated_text, skip_special_tokens=True)

    text_box[3] = translated_sentence

    pbar.update(1)
    return text_box


def translate_text_box_array(final_text_box_array):
    print_with_timestamp("Start translate_text_box()")

    try:
        # Choose a model for Japanese to English translation
        model_name = "Helsinki-NLP/opus-mt-ja-en"

        # Download the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        model = model.to(get_device())

        # Initialize concurrent futures executor
        num_workers = 4  # Number of parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            translated_text_boxes = []

            # Create a tqdm progress bar to track the number of translated lines
            with tqdm(total=len(final_text_box_array), unit="text_box", position=0, leave=True) as pbar:
                # Use ThreadPoolExecutor.map to process tasks in order
                translated_text_boxes = list(executor.map(lambda text_box: translate_text_box(text_box, tokenizer, model, get_device(), pbar),
                                                     final_text_box_array))

        # Replace the japanese lines with the translated lines
        final_text_box_array = translated_text_boxes
    except Exception as e:
        print_with_timestamp(f"Error - translate_text_box_array(): {e}")
        traceback.print_exc()
        raise e

    print_with_timestamp("End translate_text_box_array")
    return final_text_box_array


def keep_first_three_occurrences(text):
    # Split the text into words
    words = re.findall(r'\S+', text)

    # Keep track of word occurrences
    occurrences = {}
    result_words = []

    for word in words:
        word_lower = word.lower()
        occurrences.setdefault(word_lower, 0)

        # Keep the word if occurrences are less than 3
        if occurrences[word_lower] < 3:
            result_words.append(word)
            occurrences[word_lower] += 1

    return ' '.join(result_words)


def ocr_video(video_file_path):
    print_with_timestamp("Start ocr_video(" + video_file_path + ")")

    # Reader output holding a frame number, text box coordinates, text
    reader_output = []

    # Setup EasyOCR Reader
    print_with_timestamp('Setup EasyOCR')
    reader = easyocr.Reader(['ja'])  # Specify language(s)

    # Setup MangaOcr Reader
    print_with_timestamp('Setup MangaOCR')
    mocr = MangaOcr()

    # Load video file to cv2
    print_with_timestamp('Start OpenCV VideoCapture')
    cap = cv2.VideoCapture(video_file_path, cv2.CAP_FFMPEG)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(f'fps: {fps}')

    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print(f'width:{width}')
    #print(f'height:{height}')

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize a progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    # Loop through each frame from video
    i = 0  # frame number
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            # Make sure the frame exists
            if ret:
                if i % 3 == 0:  # process every 3 frames
                    # Grayscale the image for easier processing
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Run EasyOCR on frame
                    results = reader.readtext(gray_frame, paragraph=False, min_size=200)
                    frame_result_array = []
                    # Loop through each text box on the frame, and run MangaOCR on it to get better text
                    for result in results:
                        box = result[0]
                        text = result[1]
                        confidence = result[2]

                        # Only care about boxes that have text that's longer than 2 characters
                        if len(text) > 0:
                            # Convert the box coordinates to a NumPy array
                            np_box = np.array(box)

                            # Convert to integers
                            np_box = np_box.astype(int)

                            # Crop the frame to the specified box
                            cropped_frame = gray_frame[min(np_box[:, 1]):max(np_box[:, 1]), min(np_box[:, 0]):max(np_box[:, 0])]

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
            else:
                # Once the video is finished break out of the loop
                break
    except Exception as e:
        print_with_timestamp(f"Error: {e}")
        traceback.print_exc()
    finally:
        # close video and all cv2 instances
        cap.release()
        cv2.destroyAllWindows()

    print_with_timestamp('End OpenCV VideoCapture')

    # Loop through all of the reader_output to find the start/end frames of each text box
    # Use the frame number to see if we've jumped ahead in the video, thus having a new text box
    # Then use the box location and size to first see if we are using the same text
    # Then use JellyFish to compare text to see how similar they are
    # If it's the same box and text keep going, counting up the frames till you get to the end frame
    print_with_timestamp('Format captured text_boxes')
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
                active_text = [frame_result[0], frame_result[0], frame_result[1], Counter([frame_result[2]])]
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
                if frame_result_array is not None and len(frame_result_array) > 0 and len(frame_result_array[0]) > 0 and frame_result_array[0][0] - active_text[1] > (MAX_FRAME_JUMP_SEC * fps):
                    #print_with_timestamp(f'MAX_FRAME_JUMP_SEC - {active_text}')
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
                        if jf.jaro_similarity(active_text[3].most_common(1)[0][0], frame_result[2]) > JELLYFISH_ACCURACY:
                            # Add to text count
                            active_text[3].update([frame_result[2]])
                            active_text_array[i][3].update([frame_result[2]])
                            # Update end frame
                            active_text[1] = frame_result[0]
                            active_text_array[i][1] = frame_result[0]
                            # Remove frame_result from array so we don't need to look at it again
                            frame_result_array.pop(j)
                            break

            # Loop through remaining frame_resutls and add them to the active_text_array
            # (all text boxes that are continuing from last frame have already been popped off)
            #if len(frame_result_array) > 0:
            #    print_with_timestamp(f'add remaining frame_results: {frame_result_array}')
            for j in range(len(frame_result_array) - 1, -1, -1):
                frame_result = frame_result_array[j]
                # start_frame, end_frame, box, text
                active_text = [frame_result[0], frame_result[0], frame_result[1], Counter([frame_result[2]])]
                active_text_array.append(active_text)

    # Loop through remaining active_texts and add them to final_text_botx_array
    for i in range(len(active_text_array) - 1, -1, -1):
        final_text_box_array.append(active_text_array.pop(i))

    # Loop through final_text_box_array and remove all MIN_FRAME_LENGTH_SEC frame text boxes
    for i in range(len(final_text_box_array) - 1, -1, -1):
        if final_text_box_array[i][1] - final_text_box_array[i][0] <= (MIN_FRAME_LENGTH_SEC * fps):
            final_text_box_array.pop(i)

    # Sort the array based on the first number in each subarray
    final_text_box_array = sorted(final_text_box_array, key=lambda x: x[0])

    # Loop through final_text_box_array and update the frame start/end times to actual time
    for text_box in final_text_box_array:
        text_box[0] = frames_to_time(text_box[0], fps)
        text_box[1] = frames_to_time(text_box[1], fps)

    # Loop through final_text_box_array and only use the most common text:
    for i in range(len(final_text_box_array)):
        final_text_box_array[i][3] = final_text_box_array[i][3].most_common(1)[0][0]

    # Translate final_text_box_array
    final_text_box_array = translate_text_box_array(final_text_box_array)

    # Loop through english and remove unessesary entries
    for i in range(len(final_text_box_array) - 1, -1, -1):
        text_box = final_text_box_array[i]
        if ((text_box[3].lower() == 'I don\'t know.'.lower() or
                text_box[3].lower() == 'Huh.'.lower() or
                text_box[3].lower() == 'Hmm.'.lower() or
                text_box[3].lower() == 'Mm - hmm.'.lower() or
                text_box[3].lower() == 'Mm-hmm.'.lower() or
                text_box[3].lower() == '*'.lower()) or
                'I don\'t know what I\'m talking about.' in text_box[3]):
            final_text_box_array.pop(i)
        elif len(text_box[3]) > 9:
            final_text_box_array[i][3] = keep_first_three_occurrences(text_box[3])

    '''
    for text_box in final_text_box_array:
        print(text_box)
    '''

    print_with_timestamp('Write to .ass subtitle file')

    header = """[Script Info]
; Script generated by Python
Title: My Subtitle File
ScriptType: v4.00
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Text_Box,Helvetica,10,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Loop through final_text_box_array and create an .srt file
    directory_path = os.path.dirname(video_file_path)
    en_subtitle_path = (directory_path + "/" + os.path.splitext(os.path.basename(video_file_path))[0]
                        + ".textbox-en.ass")
    create_empty_srt_file(en_subtitle_path)
    # Write translated lines to the output SRT file
    with open(en_subtitle_path, 'w', encoding='utf-8') as f:
        f.write(header)

        for i in range(len(final_text_box_array)):
            text_box = final_text_box_array[i]
            start_time, end_time, box_coordinates, text = text_box
            # Take the center of the X and Y coordinates (x1+x2/2), and scale down by 3.5
            x_coordinate = (box_coordinates[0][0] + box_coordinates[1][0]) / 2 / 3.5
            y_coordinate = (box_coordinates[0][1] + box_coordinates[2][1]) / 2 / 3.5
            # Make sure the X coordinate is always at least 5% from the right so it doesn't go off screen
            x_coordinate = min(x_coordinate, (width /3.5 * 0.95))
            # Make sure the Y coordinate is always at least 15% from the bottom so it doesn't hit normal subtitles
            y_coordinate = min(y_coordinate, (height / 3.5 * 0.75))
            box_position = f"{x_coordinate},{y_coordinate}"
            f.write(f'Dialogue: 0,{start_time},{end_time},Text_Box,,0,0,0,,{{\\pos({box_position})}}{text}\n')


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
