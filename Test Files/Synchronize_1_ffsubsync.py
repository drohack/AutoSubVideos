import os
import tkinter
import datetime
import subprocess
from tkinter import filedialog


def print_with_timestamp(message):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(formatted_time, message)


def create_empty_srt_file(file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("")


def synchronize_subtitles(video_file, input_srt_file, output_srt_file):
    print("Start synchronize_subtitles(" +
          video_file + "," +
          input_srt_file + "," +
          output_srt_file + ")")
    try:
        # Construct the FFSubSync command
        ffsubsync_command = [
            "ffsubsync",
            "-a",
            video_file,
            "--vad", "webrtc",
            "-i", input_srt_file,
            "-o", output_srt_file
        ]

        # Run FFSubSync using subprocess
        subprocess.run(ffsubsync_command, check=True)

        print("Synchronization complete. Synchronized subtitles saved:", output_srt_file)
    except subprocess.CalledProcessError as e:
        print("Error:", e)


# Define the path to the video file
tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
video_path = filedialog.askopenfilename()

print_with_timestamp("Synchronizing video file: " + video_path)

# Define the path to the subtitle file
tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
subtitle_path = filedialog.askopenfilename()

print_with_timestamp("Synchronizing .srt file: " + subtitle_path)

try:
    # Synchronize subtitle file with video
    directory_path = os.path.dirname(subtitle_path)
    output_subtitle_path = directory_path + "/" + os.path.basename(subtitle_path).rstrip(".srt") + ".sync.srt"
    create_empty_srt_file(output_subtitle_path)
    synchronize_subtitles(
        video_file=video_path,
        input_srt_file=subtitle_path,
        output_srt_file=output_subtitle_path
    )
finally:
    print_with_timestamp("Done")
