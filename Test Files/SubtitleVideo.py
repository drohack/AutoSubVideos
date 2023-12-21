import os
import tkinter
import tempfile
import shutil
import subprocess
import vosk
from tqdm import tqdm
import ffmpeg
import wave
import json
from translate import Translator
from tkinter import filedialog
from moviepy.editor import VideoFileClip

'''
def transcribe_audio_with_google(input_file, output_file):
    print("Start transcribe_audio_with_google(" + input_file + "," + output_file + ")")
    recognizer = sr.Recognizer()

    with sr.AudioFile(input_file) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language="ja-JP")  # Set the language to Japanese (ja-JP)
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(text)
        print("Transcription complete. Text saved to", output_file)
        return True
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return False
    except sr.RequestError as e:
        print("Could not request results from Google Web Speech API; {0}".format(e))
        return False


def text_to_srt(input_text_file, output_srt_file):
    print("Start text_to_srt(" + input_text_file + "," + output_srt_file + ")")
    subs = pysrt.SubRipFile()

    with open(input_text_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    start_time = 0
    for line in lines:
        end_time = start_time + 1000  # 1 second per line (adjust as needed)
        subs.append(pysrt.SubRipItem(index=len(subs) + 1, start=start_time, end=end_time, text=line.strip()))
        start_time = end_time

    subs.save(output_srt_file, encoding="utf-8")
    print("Subtitle file created:", output_srt_file)


def synchronize_subtitles(video_file, input_srt_file, output_srt_file):
    print("Start synchronize_subtitles(" +
          video_file + "," +
          input_srt_file + "," +
          output_srt_file + ")")
    try:
        # Construct the FFSubSync command
        ffsubsync_command = [
            "ffsubsync",
            video_file,
            "-i", input_srt_file,
            "-o", output_srt_file
        ]

        # Run FFSubSync using subprocess
        subprocess.run(ffsubsync_command, check=True)

        print("Synchronization complete. Synchronized subtitles saved:", output_srt_file)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
'''


def transcribe_audio(audio_file_path, sub_path):
    print("Start transcribe_audio(" + audio_file_path + "," + sub_path + ")")
    sample_rate = 16000

    # Initialize Vosk recognizer
    vosk_model_path = "../vosk-model-ja-0.22"
    model = vosk.Model(vosk_model_path)
    recognizer = vosk.KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)

    # Use ffmpeg to sync audio to subtitle file, and use vosk to transcribe audio to .srt file
    with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
                           audio_file_path,
                           "-ar", str(sample_rate), "-ac", "1",
                           "-f", "s16le", "-threads", "0", "-"],
                          stdout=subprocess.PIPE).stdout as stream:
        with open(sub_path, "w", encoding="utf-8") as fh:
            fh.write(recognizer.SrtResult(stream))



def create_empty_srt_file(file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("")


def translate_subtitle(subtitle_file, output_file_path):
    print("Start translate_subtitle(" + subtitle_file + "," + output_file_path + ")")
    # Read subtitle file
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        subtitle_lines = file.readlines()
        file.close()

    # Translate subtitle text
    translator = Translator(from_lang='ja', to_lang='en')

    # Translate and write the output SRT file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for line in subtitle_lines:
            if '-->' in line:
                f.write(line)  # Keep the timestamp line unchanged
            else:
                translated_text = translator.translate(line)
                f.write(translated_text + '\n')
        f.close()


# Define the path to the video file
tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
video_path = filedialog.askopenfilename()

print("Creating subtitle file for: " + video_path)

# Create a temporary directories
temp_dir = tempfile.mkdtemp()
#temp_dir2 = tempfile.mkdtemp()

# Define the path for the temporary audio file
temp_audio_file_path = os.path.join(temp_dir, "temp_audio.wav")

# Load the video clip
video_clip = VideoFileClip(video_path)

# Extract audio and save it as a temporary audio file
audio_clip = video_clip.audio
try:
    ffmpeg_params = ["-ac", "1"]
    audio_clip.write_audiofile(temp_audio_file_path, ffmpeg_params=ffmpeg_params)

    print("Temporary audio file saved:", temp_audio_file_path)

    '''
    # Perform speech-to-text on the audio using speech_recognition with Google Text To Speech
    temp_stt_file = tempfile.NamedTemporaryFile(suffix=".srt", delete=False, dir=temp_dir)
    temp_stt_file_path = temp_stt_file.name
    if transcribe_audio_with_google(temp_audio_file_path, temp_stt_file_path):
    
        # Generate the subtitle file using pysrt
        # Save the .srt file to the same folder that the video was in
        temp_srt_file = tempfile.NamedTemporaryFile(suffix=".srt", delete=False, dir=temp_dir2)
        temp_srt_file_path = temp_srt_file.name
        text_to_srt(temp_stt_file_path, temp_srt_file_path)
    
        # Synchronize subtitle file with video
        directory_path = os.path.dirname(video_path)
        subtitle_path = os.path.splitext(os.path.basename(video_path))[0] + ".srt"
        synchronize_subtitles(
            video_file=video_path,
            input_srt_file=temp_srt_file_path,
            transcribed_audio_file=temp_stt_file_path,
            output_srt_file=subtitle_path
        )
    '''

    # Generate .srt file from audio file
    directory_path = os.path.dirname(video_path)
    jp_subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(video_path))[0] + ".jp.srt"
    create_empty_srt_file(jp_subtitle_path)
    transcribe_audio(temp_audio_file_path, jp_subtitle_path)

    '''
    # Synchronize subtitle file with video
    directory_path = os.path.dirname(video_path)
    subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(video_path))[0] + ".srt"
    create_empty_srt_file(subtitle_path)
    synchronize_subtitles(
        video_file=video_path,
        input_srt_file=temp_srt_file_path,
        output_srt_file=subtitle_path
    )
    '''

    '''
    en_subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(video_path))[0] + ".en.srt"
    create_empty_srt_file(en_subtitle_path)
    translate_subtitle(jp_subtitle_path, en_subtitle_path)
    '''
finally:
    # Close the video and audio clips
    video_clip.close()
    audio_clip.close()

    # Clean up: Delete the temporary directory and its contents
    shutil.rmtree(temp_dir)
    #shutil.rmtree(temp_dir2)

    print("done")
