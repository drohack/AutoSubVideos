# This application will take a vvideo file with japanese audio, and create 2 subtitle files:
#   1. Japanese subtitle that is synced with the audio
#   2. English translated subtitle file

# This application uses faster_whisper, ffsubsync, and Helsinki-NLP/opus-mt-ja-en to perform
# the transcription, syncing, and translation

import os
import tkinter
import tempfile
import shutil
import datetime
import subprocess
import concurrent.futures
import torch
import traceback
import platform
import sys
from datetime import timedelta
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from tkinter import filedialog
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel

import re
import math
from collections import Counter
import easyocr  # https://www.jaided.ai/easyocr/documentation/
import cv2
import numpy as np
from PIL import Image
from manga_ocr import MangaOcr  # https://github.com/kha-white/manga-ocr
import jellyfish as jf


RESULT_CONFIDENCE = 0.0
JELLYFISH_ACCURACY = 0.5        # From 0 to 1, how close 2 lines of text have to be to be considered the same string
BOX_DISTANCE_REQUIREMENT = 30   # Maximum center distance of 2 boxes need to be to be considered the same box
BOX_SIZE_REQUIREMENT = 90       # Maximum area difference of 2 boxes need to be to be considered the same box
MAX_FRAME_JUMP_SEC = 3          # Maximum frames we wait till ending a text box (seconds * fps)
MIN_FRAME_LENGTH_SEC = 0.7      # Minimum duration we keep text boxes (we throw out what's lower than this) (seconds * fps)

device = "cpu"
ffsubsync_path = "ffsubsync"
width = 1920
height = 1080


def set_device(value: str):
    global device
    device = value


def get_device() -> str:
    global device
    return device


def set_ffsubsync_path(value: str):
    global ffsubsync_path
    ffsubsync_path = value


def get_ffsubsync_path() -> str:
    global ffsubsync_path
    return ffsubsync_path


def set_width(value: int):
    global width
    width = value


def get_width() -> int:
    global width
    return width


def set_height(value: int):
    global height
    height = value


def get_height() -> int:
    global height
    return height


def print_with_timestamp(message: str):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(formatted_time, message)


def check_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        result.check_returncode()  # This will raise a CalledProcessError if the exit code is non-zero
    except subprocess.CalledProcessError as ex:
        print_with_timestamp(f"Error running ffmpeg: {ex}")
        return False
    return True


def install_ffmpeg():
    system = platform.system()

    if system == "Windows":
        # Install FFmpeg on Windows
        subprocess.run(["pip", "install", "imageio[ffmpeg]"], check=True)
    elif system == "Linux":
        # Install FFmpeg on Linux
        subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"], check=True)
    else:
        print_with_timestamp("Unsupported operating system to install ffmpeg")
        sys.exit(1)



def check_ffsubsync():
    try:
        # Try to run FFSubSync and check the version
        result = subprocess.run([get_ffsubsync_path(), "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        result.check_returncode()  # This will raise a CalledProcessError if the exit code is non-zero
    except subprocess.CalledProcessError as ex:
        print_with_timestamp(f"Error running ffsubsync: {ex}")
        return False
    except Exception as ex:
        print_with_timestamp(f"Exception running ffsubsync: {ex}")
        return False
    return True


def get_ffsubsync_scripts_path():
    try:
        # Run pip show ffsubsync command and capture the output
        result = subprocess.run(['pip', 'show', 'ffsubsync'], check=True, capture_output=True, text=True)

        # Extract the location of the 'Scripts' directory from the output
        scripts_path = None
        for line in result.stdout.splitlines():
            if line.startswith('Location:'):
                _, scripts_path = line.split(':', 1)
                scripts_path = scripts_path.strip()
                break

        return scripts_path
    except subprocess.CalledProcessError as e:
        print_with_timestamp(f"Error getting ffsubsync location: {e}")
        return None


def install_ffsubsync():
    system = platform.system()

    if system == "Windows":
        # Install FFSubSync on Windows (adjust the installation command as needed)
        # subprocess.run(["pip", "install", "imageio[ffsubsync]"], check=True)
        try:
            # Use pip to install ffsubsync
            subprocess.check_call(['pip', 'install', 'ffsubsync'])

            # Get the 'Scripts' directory location of ffsubsync using pip show
            set_ffsubsync_path(get_ffsubsync_scripts_path() + '/ffsubsync/fsubsync.py')
            print_with_timestamp("ffsubsync_path: " + get_ffsubsync_path())

            print_with_timestamp("ffsubsync installed successfully!")
        except subprocess.CalledProcessError as e:
            print_with_timestamp(f"Error installing ffsubsync: {e}")
            sys.exit(1)
        except Exception as e:
            print_with_timestamp(f"An unexpected error occurred: {e}")
            sys.exit(1)
    elif system == "Linux":
        # Install FFSubSync on Linux (adjust the installation command as needed)
        subprocess.run(["sudo", "apt-get", "install", "-y", "ffsubsync"], check=True)
    else:
        print_with_timestamp("Unsupported operating system to install FFSubSync")
        sys.exit(1)



def check_gpu():
    # Check if GPU is available
    if torch.cuda.is_available():
        # Set GPU device
        torch.cuda.set_device(0)
        gpu_properties = torch.cuda.get_device_properties(0)

        # Check the CUDA version
        cuda_version = torch.version.cuda
        gpu_version = str(gpu_properties.major) + "." + str(gpu_properties.minor)
        print_with_timestamp(f"Torch CUDA version: {cuda_version}")
        print_with_timestamp(f"GPU   CUDA version: {gpu_version}")

        # Check PyTorch CUDA version compatibility
        required_cuda_version = torch.version.cuda.split('.')[0]
        if int(required_cuda_version) <= float(gpu_version):
            print_with_timestamp("Using GPU - Your GPU is compatible with PyTorch.")
        else:
            print_with_timestamp("Using GPU - Your GPU may not be compatible with this version of PyTorch.")

        set_device("cuda")
    else:
        print_with_timestamp("No GPU available. PyTorch is running on CPU.")


def transcribe_audio(video_path: str) -> str:
    print_with_timestamp("Start transcribe_audio(" + video_path + ")")

    # Create jp_subtitle_path
    directory_path = os.path.dirname(video_path)
    jp_subtitle_path = (directory_path + "/" + os.path.splitext(os.path.basename(video_path))[0]
                        + ".faster_whisper.jp.srt")

    # Create a temporary directories
    temp_dir = tempfile.mkdtemp()

    # Define the path for the temporary audio file
    temp_audio_file_path = os.path.join(temp_dir, "temp_audio.wav")

    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Extract audio and save it as a temporary audio file
    audio_clip = video_clip.audio
    try:
        ffmpeg_params = ["-ac", "1"]
        audio_clip.write_audiofile(temp_audio_file_path, ffmpeg_params=ffmpeg_params)

        print_with_timestamp("Temporary audio file saved:" + temp_audio_file_path)

        # Transcribe with faster_whisper
        # Run on GPU
        print_with_timestamp("Load Faster Whisper")
        model = WhisperModel("large-v3", device=get_device(), compute_type="auto", num_workers=5)
        segments, info = model.transcribe(audio=temp_audio_file_path, beam_size=1, language='ja', temperature=0,
                                          word_timestamps=True, condition_on_previous_text=False,
                                          no_speech_threshold=0.1,
                                          # ,vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
                                          )
        print_with_timestamp("Faster Whisper loaded")

        print_with_timestamp("Transcribe audio")
        transcribed_str = ""
        with (tqdm(total=None) as pbar):
            for segment in segments:
                # Convert seconds to timedelta
                start_delta = timedelta(seconds=segment.start)
                start_hours, start_remainder = divmod(start_delta.seconds, 3600)
                start_minutes, start_seconds = divmod(start_remainder, 60)
                start_milliseconds = start_delta.microseconds // 1000
                start_time = "{:02}:{:02}:{:02},{:03}".format(start_hours, start_minutes, start_seconds,
                                                              start_milliseconds)
                end_delta = timedelta(seconds=segment.end)
                end_hours, end_remainder = divmod(end_delta.seconds, 3600)
                end_minutes, end_seconds = divmod(end_remainder, 60)
                end_milliseconds = end_delta.microseconds // 1000
                if float(segment.start) == float(segment.end):
                    end_time = "{:02}:{:02}:{:02},{:03}".format(end_hours, end_minutes, end_seconds,
                                                                end_milliseconds + 0.5)
                else:
                    end_time = "{:02}:{:02}:{:02},{:03}".format(end_hours, end_minutes, end_seconds,
                                                                end_milliseconds)
                line = "%d\n%s --> %s\n%s\n\n" % (segment.id, start_time, end_time, segment.text)
                transcribed_str += line
                pbar.update()

        # Write transcription to .srt file
        with open(jp_subtitle_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(transcribed_str)
    except Exception as e:
        print_with_timestamp(f"Error - transcribe_audio(): {e}")
        traceback.print_exc()
        raise e
    finally:
        # Close the video and audio clips
        video_clip.close()
        audio_clip.close()

        # Clean up: Delete the temporary directory and its contents
        shutil.rmtree(temp_dir)

    print_with_timestamp("End transcribe_audio: " + jp_subtitle_path)

    return jp_subtitle_path


def create_empty_srt_file(file_path: str):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("")


def synchronize_subtitles(video_file: str, input_srt_file: str) -> str:
    print_with_timestamp("Start synchronize_subtitles(" + video_file + "," + input_srt_file + ")")

    base_path = os.path.dirname(input_srt_file)
    output_srt_file = (base_path + "/" + os.path.basename(input_srt_file).replace(".jp.srt", ".jp-sync.srt"))
    create_empty_srt_file(output_srt_file)

    print_with_timestamp("Created empty output file: " + output_srt_file)

    try:
        # Construct the FFSubSync command
        ffsubsync_command = [
            get_ffsubsync_path(),
            video_file,
            "--vad", "webrtc",
            "-i", input_srt_file,
            "-o", output_srt_file
        ]

        # Run FFSubSync using subprocess
        subprocess.run(ffsubsync_command, check=True)

        # Delete input_str_file so we only have a single japanese subtitle file
        try:
            os.remove(input_srt_file)
            print_with_timestamp(f"File '{input_srt_file}' deleted successfully.")
            os.rename(output_srt_file, input_srt_file)
            output_srt_file = input_srt_file
            print_with_timestamp(f"File '{output_srt_file}' renamed successfully.")
        except FileNotFoundError:
            print_with_timestamp(f"Error: File '{input_srt_file}' does not exist.")

        print_with_timestamp("End synchronize_subtitles: " + output_srt_file)
    except subprocess.CalledProcessError as e:
        print_with_timestamp(f"CalledProcessError - synchronize_subtitles(): {e}")
        traceback.print_exc()
        raise e
    except Exception as e:
        print_with_timestamp(f"Error - synchronize_subtitles(): {e}")
        traceback.print_exc()
        raise e

    return output_srt_file


def clean_up_line(line: str) -> str:
    line = line.replace("、", ", ")
    line = line.replace("・", " ")
    line = line.replace("…", "...")
    line = line.replace("。", ".")
    return line


def translate_line(line: str, tokenizer, model, device, pbar) -> str:
    encoded_text = tokenizer(line, return_tensors="pt")
    # Run on GPU if available
    encoded_text = encoded_text.to(device)

    translated_text = model.generate(**encoded_text)[0]
    translated_sentence = tokenizer.decode(translated_text, skip_special_tokens=True)

    pbar.update(1)
    return translated_sentence


def update_pbar(pbar):
    pbar.update(1)


def translate_subtitle_parallel(input_subtitle_path: str) -> str:
    print_with_timestamp("Start translate_subtitle_parallel(" + input_subtitle_path + ")")

    directory_path = os.path.dirname(input_subtitle_path)
    en_subtitle_path = (directory_path + "/" + os.path.splitext(os.path.basename(input_subtitle_path))[0]
                        + ".huggy.en.srt")
    create_empty_srt_file(en_subtitle_path)

    try:
        # Choose a model for Japanese to English translation
        model_name = "Helsinki-NLP/opus-mt-ja-en"

        # Download the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        model = model.to(get_device())

        # Read subtitle file
        with open(input_subtitle_path, 'r', encoding='utf-8') as file:
            subtitle_lines = file.readlines()

        # Get only the text lines, every 4th line starting at the 3rd line
        text_lines = subtitle_lines[2::4]

        # Initialize concurrent futures executor
        num_workers = 4  # Number of parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            translated_lines = []

            # Create a tqdm progress bar to track the number of translated lines
            with tqdm(total=len(text_lines), unit="line", position=0, leave=True) as pbar:
                # Use ThreadPoolExecutor.map to process tasks in order
                translated_lines = list(executor.map(lambda line: translate_line(line, tokenizer, model, get_device(), pbar),
                                                     text_lines))

        # Replace the japanese lines with the translated lines (this keeps the index, and timestamps)
        subtitle_lines[2::4] = translated_lines

        # Write translated lines to the output SRT file
        with open(en_subtitle_path, 'w', encoding='utf-8') as f:
            for line in subtitle_lines:
                if line.endswith('\n'):
                    f.write(line)
                else:
                    f.write(line + '\n')
    except Exception as e:
        print_with_timestamp(f"Error - translate_subtitle_parallel(): {e}")
        traceback.print_exc()
        raise e

    print_with_timestamp("End translate_subtitle_parallel: " + en_subtitle_path)
    return en_subtitle_path


def are_boxes_similar(box1, box2):
    # Extracting coordinates and dimensions
    x1, y1, w1, h1 = box1[0][0], box1[0][1], box1[1][0] - box1[0][0], box1[3][1] - box1[0][1]
    x2, y2, w2, h2 = box2[0][0], box2[0][1], box2[1][0] - box2[0][0], box2[3][1] - box2[0][1]

    # Calculate center points
    center1 = ((x1 + x1 + w1) // 2, (y1 + y1 + h1) // 2)
    center2 = ((x2 + x2 + w2) // 2, (y2 + y2 + h2) // 2)

    # Calculate distance between centers
    distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    # Calculate size difference
    size_difference = abs(w1 - w2) + abs(h1 - h2)

    if distance < BOX_DISTANCE_REQUIREMENT and size_difference < BOX_SIZE_REQUIREMENT:
        return True
    else:
        # print_with_timestamp(f"Distance between centers: {distance}")
        # print_with_timestamp(f"Size difference: {size_difference}")
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
                translated_text_boxes = list(
                    executor.map(lambda text_box: translate_text_box(text_box, tokenizer, model, get_device(), pbar),
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
    # print_with_timestamp(f'fps: {fps}')

    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    set_width(width)
    set_height(height)
    # print_with_timestamp(f'width:{width}')
    # print_with_timestamp(f'height:{height}')

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
                            cropped_frame = gray_frame[min(np_box[:, 1]):max(np_box[:, 1]),
                                            min(np_box[:, 0]):max(np_box[:, 0])]

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
                        # print_with_timestamp(frame_result_array)

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
        # print_with_timestamp(f'____frame____: {frame_result_array[0][0]}')

        # If there's no active text, then add the next frame_results as active text setting their starting
        # and ending frame as the existing frame
        # Else compare the frame_resutls to any active text
        if len(active_text_array) == 0:
            for i in range(len(frame_result_array) - 1, -1, -1):
                frame_result = frame_result_array[i]
                # start_frame, end_frame, box, text
                active_text = [frame_result[0], frame_result[0], frame_result[1], Counter([frame_result[2]])]
                active_text_array.append(active_text)

            # print_with_timestamp(f'initialize active_text_array: {active_text_array}')
            # Go to next frame_result_array
            continue
        else:
            # Compare each active_text against each frame_result
            for i in range(len(active_text_array) - 1, -1, -1):
                active_text = active_text_array[i]
                new_active_end_frame = active_text[1]

                # If we jumped too many frames, move the active_text to final_text_box_array,
                # and skip to the next active_text
                if frame_result_array is not None and len(frame_result_array) > 0 and len(frame_result_array[0]) > 0 and \
                        frame_result_array[0][0] - active_text[1] > (MAX_FRAME_JUMP_SEC * fps):
                    # print_with_timestamp(f'MAX_FRAME_JUMP_SEC - {active_text}')
                    final_text_box_array.append(active_text_array.pop(i))
                    continue

                # Loop through all frame_results and compare them to the active_text
                for j in range(len(frame_result_array) - 1, -1, -1):
                    frame_result = frame_result_array[j]
                    # Check if text boxes locations are similar
                    # print_with_timestamp(f'are_boxes_similar: {are_boxes_similar(active_text[2], frame_result[1])}')
                    if are_boxes_similar(active_text[2], frame_result[1]):
                        # Check if the text is similar using JellyFish
                        # print_with_timestamp(f'jaro_similarity: {jf.jaro_similarity(active_text[3], frame_result[2])}')
                        if jf.jaro_similarity(active_text[3].most_common(1)[0][0],
                                              frame_result[2]) > JELLYFISH_ACCURACY:
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
            # if len(frame_result_array) > 0:
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
        if (text_box[3].lower() == 'I don\'t know.'.lower() or
            text_box[3].lower() == 'Huh.'.lower() or
            text_box[3].lower() == 'Hmm.'.lower() or
            text_box[3].lower() == 'Mm - hmm.'.lower() or
            text_box[3].lower() == 'Mm-hmm.'.lower() or
            text_box[3].lower() == '*'.lower() or
            text_box[3].lower() == '.'.lower() or
            'I don\'t know what I\'m talking about.' in text_box[3]):
            final_text_box_array.pop(i)
        elif len(text_box[3]) > 9:
            final_text_box_array[i][3] = keep_first_three_occurrences(text_box[3])

    '''
    for text_box in final_text_box_array:
        print_with_timestamp(text_box)
    '''

    print_with_timestamp("End ocr_video()")
    return final_text_box_array


def combine_and_write_to_ass_file(video_file_path, sub_file_path, ocr_data):
    print_with_timestamp("Start combine_and_write_to_ass_file(" + video_file_path + ", " + sub_file_path + ", ocr_data)")

    # Read subtitle file
    with open(sub_file_path, 'r', encoding='utf-8') as file:
        subtitle_lines = file.readlines()

    # Convert to ocr_data format [start_time, end_time, box, text]
    srt_data = []
    i = 1  # skip first line
    while i < len(subtitle_lines):
        #print_with_timestamp(subtitle_lines[i])
        # Get start and end time
        time = subtitle_lines[i].replace('\n', '')
        times = time.split(" --> ")
        start_time = times[0].replace(',', '.')[:-1]
        end_time = times[1].replace(',', '.')[:-1]
        # Go to next line
        i += 1
        text = subtitle_lines[i]
        # create subtitle_data
        data = [start_time, end_time, None, text]
        srt_data.append(data)
        # skip the blank line and index to go to the next line with time
        i += 3

    # Combine the srt_data and ocr_data
    combined_subtitle_data = srt_data + ocr_data

    # Sort the array based on the first item of each sub-array
    subtitle_data = sorted(combined_subtitle_data, key=lambda x: x[0])

    header = """[Script Info]
; Script generated by Python
Title: My Subtitle File
ScriptType: v4.00
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Helvetica,16,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Text_Box,Helvetica,10,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Loop through final_text_box_array and create an .srt file
    directory_path = os.path.dirname(video_file_path)
    final_subtitle_path = (directory_path + "/" + os.path.splitext(os.path.basename(video_file_path))[0]
                           + ".final.en-ai.ass")
    create_empty_srt_file(final_subtitle_path)
    # Write translated lines to the output SRT file
    with open(final_subtitle_path, 'w', encoding='utf-8') as f:
        f.write(header)

        for i in range(len(subtitle_data)):
            data = subtitle_data[i]
            start_time, end_time, box_coordinates, text = data
            if box_coordinates is not None:
                # Take the center of the X and Y coordinates (x1+x2/2), and scale down by 3.5
                x_coordinate = (box_coordinates[0][0] + box_coordinates[1][0]) / 2 / 3.5
                y_coordinate = (box_coordinates[0][1] + box_coordinates[2][1]) / 2 / 3.5
                # Make sure the X coordinate is always at least 5% from the right so it doesn't go off screen
                x_coordinate = min(x_coordinate, (get_width() / 3.5 * 0.95))
                # Make sure the Y coordinate is always at least 15% from the bottom so it doesn't hit normal subtitles
                y_coordinate = min(y_coordinate, (get_height() / 3.5 * 0.75))
                box_position = f"{{\\pos({x_coordinate},{y_coordinate})}}"
                f.write(f'Dialogue: 0,{start_time},{end_time},Text_Box,,0,0,0,,{box_position}{text}\n')
            else:
                new_line = f'Dialogue: 1,{start_time},{end_time},Default,,0,0,0,,{text}'
                if not new_line.endswith('\n'):
                    # If not, add '\n' to the end of the string
                    new_line += '\n'
                f.write(new_line)

    print_with_timestamp(f"End combine_and_write_to_ass_file({final_subtitle_path})")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_with_timestamp("Create Subtitle File")

    # Check if FFmpeg is installed
    print_with_timestamp("Checking if ffmpeg is already installed.")
    if not check_ffmpeg():
        print_with_timestamp("FFmpeg not found. Installing...")
        install_ffmpeg()
        # Check again after installation
        if check_ffmpeg():
            print_with_timestamp("FFmpeg installed successfully, continuing application.")
        else:
            print_with_timestamp("Failed to install FFmpeg. Please install it manually.")
            sys.exit(1)
    else:
        print_with_timestamp("FFmpeg is already installed, continuing application.")


    # Check if FFsubsync is installed
    print_with_timestamp("Checking if ffsubsync is already installed.")
    if not check_ffsubsync():
        print_with_timestamp("FFsubsync not found. Installing...")
        install_ffsubsync()
        # Check again after installation
        if check_ffsubsync():
            print_with_timestamp("FFsubsync installed successfully, continuing application.")
        else:
            print_with_timestamp("Failed to install FFsubsync. Please install it manually.")
            sys.exit(1)
    else:
        print_with_timestamp("FFsubsync is already installed, continuing application.")


    # Check if the GPU is available and compatible with the version of Torch, else set to run on CPU
    check_gpu()

    # Define the path to the video file
    root = tkinter.Tk()
    root.withdraw()  # prevents an empty tkinter window from appearing
    paths = filedialog.askopenfilenames(
        title="Select video file(s)",
        filetypes=[("Video files", ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mpeg', '.mpg', 'm4v'))],
    )

    # Check to see if the video file or path exists
    if paths:
        for video_path in paths:
            print_with_timestamp("Creating subtitle file for: " + video_path)

            try:
                # Generate Japanese .srt file from audio file
                jp_subtitle_path = transcribe_audio(video_path)

                # Synchronize subtitle file with video
                synced_srt_path = synchronize_subtitles(video_file=video_path, input_srt_file=jp_subtitle_path)

                # Translate the synchronized subtitle file to English
                en_subtitle_path = translate_subtitle_parallel(synced_srt_path)

                # Get the OCR object to merge into the final subtitle file
                ocr_data = ocr_video(video_path)

                # Combine subtitle and ocr data, and write to Final ass subtitle file
                combine_and_write_to_ass_file(video_path, en_subtitle_path, ocr_data)

                print_with_timestamp("Create Subtitle File - Complete")
            except Exception as e:
                print_with_timestamp(f"Create Subtitle File - Failed: {e}")
    else:
        print_with_timestamp("No file or directory selected.")


