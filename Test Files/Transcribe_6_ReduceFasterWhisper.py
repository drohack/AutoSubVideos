import os
import tkinter
import tempfile
import shutil
import datetime
from tqdm import tqdm
from tkinter import filedialog
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel

import torch
from df.enhance import enhance, init_df, load_audio, save_audio  #https://github.com/Rikorose/DeepFilterNet


def print_with_timestamp(message):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(formatted_time, message)


def transcribe_audio(audio_file_path):
    print_with_timestamp("Start transcribe_audio(" + audio_file_path + ")")
    '''
    # Transcribe with the default Whisper package
    model = whisper.load_model("large-v2")  # Change this to your desired model
    transcribe = model.transcribe(audio=audio_file_path, language='ja', fp16=False, verbose=False)
    segments = transcribe['segments']
    print_with_timestamp("End whisper")
    
    transcribed_str = ""
    with tqdm(total=len(segments)) as pbar:
        for segment in segments:
            start_time = str(0) + str(datetime.timedelta(seconds=int(segment['start']))) + ',000'
            end_time = str(0) + str(datetime.timedelta(seconds=int(segment['end']))) + ',000'
            text = segment['text']
            segment_id = segment['id'] + 1
            line = f"{segment_id}\n{start_time} --> {end_time}\n{text[1:] if text[0] == ' ' else text}\n\n"
            transcribed_str += line
            pbar.update()
    '''
    # Transcribe with faster_whisper
    # Run on GPU with INT8_FP16
    model = WhisperModel("large-v3", device="cuda", compute_type="auto", num_workers=5)
    # or run on GPU with INT8
    # model = WhisperModel("large-v2", device="cuda", compute_type="int8")
    # or run on CPU with INT8
    # model = WhisperModel("large-v2", device="cpu", compute_type="int8")
    # segments, info = model.transcribe(audio=audio_file_path, beam_size=5, language='ja', temperature=0)
    segments, info = model.transcribe(audio=audio_file_path, beam_size=1, language='ja', temperature=0,
                                      word_timestamps=True, condition_on_previous_text=False,
                                      no_speech_threshold=0.1
                                      )
    print_with_timestamp("End Load whisper")

    #print(torch.cuda.is_available())

    print_with_timestamp("Start transcribe")
    transcribed_str = ""
    with tqdm(total=None) as pbar:
        for segment in segments:
            total_seconds = int(segment.start)
            h, r = divmod(total_seconds, 3600)
            m, s = divmod(r, 60)
            ms = int(float(segment.start) * 1000) % 1000
            start_time = f"{h:02}:{m:02}:{s:02},{ms:03}"
            if float(segment.start) == float(segment.end):
                segment.end += 0.5
            total_seconds = int(segment.end)
            h, r = divmod(total_seconds, 3600)
            m, s = divmod(r, 60)
            ms = int(float(segment.start) * 1000) % 1000
            end_time = f"{h:02}:{m:02}:{s:02},{ms:03}"
            line = "%d\n%s --> %s\n%s\n\n" % (segment.id, start_time, end_time, segment.text)
            transcribed_str += line
            pbar.update()

    print_with_timestamp("End transcribe_audio()")
    return transcribed_str


def create_empty_srt_file(file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("")


# Define the path to the video file
tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
video_path = filedialog.askopenfilename()

print_with_timestamp("Creating subtitle file for: " + video_path)

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
    # Set the desired sample rate (fps) to 48000
    desired_sample_rate = 48000

    # Resample the audio to the desired sample rate
    resampled_audio = audio_clip.set_frame_rate(desired_sample_rate)

    ffmpeg_params = ["-ac", "1"]
    resampled_audio.write_audiofile(temp_audio_file_path, ffmpeg_params=ffmpeg_params)

    print_with_timestamp("Temporary audio file saved:" + temp_audio_file_path)

    print_with_timestamp("Reduce audio start")
    reduce_audio_file_path = os.path.join(temp_dir, "temp_reduced_audio.wav")
    df_model, df_state, _ = init_df()
    audio, _ = load_audio(temp_audio_file_path, sr=df_state.sr())
    enhanced = enhance(df_model, df_state, audio, atten_lim_db=6)
    # Save for listening
    save_audio(reduce_audio_file_path, enhanced, df_state.sr())
    # Clear GPU memory
    torch.cuda.empty_cache()
    print_with_timestamp("Reduce audio end")

    # Generate .srt file from audio file
    transcribed_audio = transcribe_audio(reduce_audio_file_path)

    # Write transcription to .srt file
    directory_path = os.path.dirname(video_path)
    jp_subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(video_path))[0] + ".faster_whisper.test-jp.srt"
    with open(jp_subtitle_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(transcribed_audio)

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

    print_with_timestamp("done")
