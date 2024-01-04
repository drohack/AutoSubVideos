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


device = "cpu"
ffsubsync_path = "ffsubsync"


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
        print("Unsupported operating system to install ffmpeg")
        sys.exit(1)


'''
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
'''


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_with_timestamp("Create Subtitle File")

    # Check if FFmpeg is installed
    print_with_timestamp("Checking if ffmpeg is already installed.")
    if not check_ffmpeg():
        print("FFmpeg not found. Installing...")
        install_ffmpeg()
        # Check again after installation
        if check_ffmpeg():
            print("FFmpeg installed successfully, continuing application.")
        else:
            print("Failed to install FFmpeg. Please install it manually.")
            sys.exit(1)
    else:
        print("FFmpeg is already installed, continuing application.")

    '''
    # Check if FFsubsync is installed
    print_with_timestamp("Checking if ffsubsync is already installed.")
    if not check_ffsubsync():
        print("FFsubsync not found. Installing...")
        install_ffsubsync()
        # Check again after installation
        if check_ffsubsync():
            print("FFsubsync installed successfully, continuing application.")
        else:
            print("Failed to install FFsubsync. Please install it manually.")
            sys.exit(1)
    else:
        print("FFsubsync is already installed, continuing application.")
    '''

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
                # synced_srt_path = synchronize_subtitles(video_file=video_path, input_srt_file=jp_subtitle_path)

                # Translate the synchronized subtitle file to English
                en_subtitle_path = translate_subtitle_parallel(jp_subtitle_path)

                print_with_timestamp("Create Subtitle File - Complete")
            except Exception as e:
                print_with_timestamp(f"Create Subtitle File - Failed: {e}")
    else:
        print("No file or directory selected.")


