import os
import tkinter
import tempfile
import shutil
import datetime
import torch
from tqdm import tqdm
from tkinter import filedialog
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio


def print_with_timestamp(message):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(formatted_time, message)


def transcribe_audio(audio_file_path):
    print_with_timestamp("Start transcribe_audio(" + audio_file_path + ")")

    # Choose and download model
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

    # Load audio using torchaudio
    audio, orig_freq = torchaudio.load(audio_file_path, format="wav")

    print("audio length:", audio.shape[0])
    print("orig_freq", orig_freq)
    print("audio:", audio)

    # Preprocess audio for model input
    audio = audio.squeeze(0)  # Add batch dimension
    print("audio squeeze length:", audio.shape[0])

    # Set model and audio to use GPU
    model.to("cuda")
    audio.to("cuda")

    print_with_timestamp("Facebook loaded")

    num_chunks = 100  # Adjust this as needed
    segments = torch.chunk(audio, num_chunks, dim=0)  # Specify the dimension to split along (0 for time axis)
    print("segments size:", len(segments))

    transcripts = []
    #with tqdm(total=len(segments)) as pbar:
        #for segment in segments:
    audio_inputs = processor(audios=segments[0], return_tensors="pt")
    audio_inputs.to("cuda")
    #segment_transcript = model.generate(**audio_inputs, tgt_lang="jpn")[0].cpu().numpy().squeeze()
    #transcripts.append(segment_transcript)

    output_tokens = model.generate(**audio_inputs, tgt_lang="jpn", generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    transcripts.append(translated_text_from_audio)
            #pbar.update(1)
    #final_transcript = " ".join(transcripts)
    #print(final_transcript)

    # Get the predicted text from the list of outputs
    #predicted_text = transcripts[0]

    print(transcripts[0])

    print_with_timestamp("End transcription")

    """
    transcribed_str = ""
    with tqdm(total=None) as pbar:
        for segment in segments:
            start_time = str(0) + str(datetime.timedelta(seconds=int(segment.start))) + ',000'
            end_time = str(0) + str(datetime.timedelta(seconds=int(segment.end))) + ',000'
            #text = segment.text
            #segment_id = segment.id + 1
            #line = f"{segment_id}\n{segment.start} --> {segment.end}\n{text[1:] if text[0] == ' ' else text}\n\n"
            line = "%d\n%s --> %s\n%s\n\n" % (segment.id, start_time, end_time, segment.text)
            transcribed_str += line
            pbar.update()
    """

    return ""


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
    ffmpeg_params = ["-ac", "1", "-ar", "16000"]
    audio_clip.write_audiofile(temp_audio_file_path, ffmpeg_params=ffmpeg_params)

    print_with_timestamp("Temporary audio file saved:" + temp_audio_file_path)

    # Generate .srt file from audio file
    transcribed_audio = transcribe_audio(temp_audio_file_path)

    # Write transcription to .srt file
    directory_path = os.path.dirname(video_path)
    jp_subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(video_path))[0] + ".Facebook.jp.srt"
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
