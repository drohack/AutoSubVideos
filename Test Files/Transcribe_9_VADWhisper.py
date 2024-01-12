import os
import tkinter
import tempfile
import shutil
import datetime
import traceback

from tqdm import tqdm
from tkinter import filedialog
from moviepy.editor import VideoFileClip

import stable_whisper
from stable_whisper import WhisperResult
from pydub import AudioSegment
from pydub.utils import mediainfo

from df.enhance import enhance, init_df, load_audio, save_audio  #https://github.com/Rikorose/DeepFilterNet

import torch
# https://gitlab.com/aadnk/whisper-webui
from whisper_webui.src.config import ModelConfig
from whisper_webui.src.whisper.fasterWhisperContainer import FasterWhisperContainer
from whisper_webui.app import WhisperTranscriber, VadOptions

try:
    def print_with_timestamp(message):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
        print(formatted_time, message)


    def transcribe_audio(audio_file_path, input_sr, directory_path, video_file_name):
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
        # Transcribe with stable_whisper
        '''
        model = stable_whisper.load_faster_whisper("large-v3", device="cuda", compute_type="auto", num_workers=2)
        result: WhisperResult = model.transcribe_stable(audio=audio_file_path, beam_size=1, language='ja', temperature=0,
                                                        word_timestamps=True, condition_on_previous_text=False,
                                                        no_speech_threshold=0.1,
                                                        vad_filter=True, vad_parameters=dict()
                                                        )
        '''

        try:
            jp_filename = os.path.splitext(video_file_name)[0] + ".wwebui_stable_faster_whisper.jp.srt"

            data_models = [
                        {
                            "name": "medium",
                            "url": "medium"
                        },
                        {
                            "name": "large-v3",
                            "url": "large-v3"
                        }
            ]

            models = [ModelConfig(**x) for x in data_models]

            model = FasterWhisperContainer(model_name='large-v3', device='cuda', compute_type='auto', models=models)
            model.ensure_downloaded()
            vad_options = VadOptions(vad='silero-vad', vadMergeWindow=5, vadMaxMergeSize=180,
                                     vadPadding=1, vadPromptWindow=1)
            wwebui = WhisperTranscriber()
            result = wwebui.transcribe_file(model=model, audio_path=audio_file_path, language='Japanese', task='transcribe',
                                            vadOptions=vad_options, input_sr=int(input_sr),
                                            beam_size=5,
                                            word_timestamps=True, condition_on_previous_text=True,
                                            no_speech_threshold=0.35,
                                            )

            print_with_timestamp("End whisper")

            source_download, source_text = wwebui.write_srt(result, output_filename=jp_filename, output_dir=directory_path, highlight_words=False)
        except Exception as e:
            print_with_timestamp(f"An exception occurred: {e}")
            traceback.print_exc()


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

    # Get the correct audio sample rate to extract the audio correctly
    audio_sr = mediainfo(video_path)['sample_rate']

    # Load the video's audio
    audio_clip = AudioSegment.from_file(video_path)

    # Set channels=1 to convert to mono
    mono_audio = audio_clip.set_channels(1)

    try:
        # Export the mono audio to a WAV file
        ffmpeg_params = ["-ac", "1"]
        mono_audio.export(temp_audio_file_path, format='wav', parameters=ffmpeg_params)

        print_with_timestamp("Temporary audio file saved:" + temp_audio_file_path)

        '''
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
        '''

        # Generate .srt file from audio file
        transcribe_audio(temp_audio_file_path, audio_sr, os.path.dirname(video_path), os.path.basename(video_path))

        '''
        # Write transcription to .srt file
        with open(jp_subtitle_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(transcribed_audio)
        '''
    finally:
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()

            # Close the video and audio clips
            #video_clip.close()
            #audio_clip.close()

            # Clean up: Delete the temporary directory and its contents
            shutil.rmtree(temp_dir)
            #shutil.rmtree(temp_dir2)
        except Exception as e:
            print_with_timestamp(f"An exception occurred: {e}")
            traceback.print_exc()

        print_with_timestamp("done")
except Exception as e:
    print(f"An exception occurred: {e}")
    traceback.print_exc()

print('try done')