import os
import tkinter
import tempfile
import shutil
import speech_recognition as sr
import pysrt
import subprocess
import vosk
from tqdm import tqdm
import ffmpeg
import wave
import json
import vosk_autosrt
from translate import Translator
from tkinter import filedialog
from moviepy.editor import VideoFileClip


def translate_subtitle(subtitle_file, output_file_path):
    print("Start translate_subtitle(" + subtitle_file + "," + output_file_path + ")")
    # Read subtitle file
    lines = []
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Translate subtitle text
    translator = Translator(from_lang='ja', to_lang='en')

    # Translate and write the output SRT file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        total_lines = len(lines)
        with tqdm(total=total_lines, unit="line", position=0, leave=True) as pbar:
            for line in lines:
                if '-->' in line:
                    f.write(line)  # Keep the timestamp line unchanged
                else:
                    translated_text = translator.translate(line)
                    f.write(translated_text + '\n')
                pbar.update(1)


def create_empty_srt_file(file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("")


# Define the path to the video file
tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
sub_path = filedialog.askopenfilename()

print("Translating subtitle file for: " + sub_path)

directory_path = os.path.dirname(sub_path)
en_subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(sub_path))[0] + ".en.srt"
create_empty_srt_file(en_subtitle_path)
translate_subtitle(sub_path, en_subtitle_path)
