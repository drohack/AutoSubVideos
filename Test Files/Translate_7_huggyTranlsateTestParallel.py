import os
import tkinter
import concurrent.futures
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from tkinter import filedialog
import pysrt


def clean_up_line(line: str) -> str:
    line = line.replace("、", ", ")
    line = line.replace("・", " ")
    line = line.replace("…", "...")
    line = line.replace("。", ".")
    return line


def translate_line(sub, tokenizer, model, device, pbar) -> str:
    encoded_text = tokenizer(sub.text, return_tensors="pt")
    # Run on GPU if available
    encoded_text = encoded_text.to(device)

    translated_text = model.generate(**encoded_text)[0]
    translated_sentence = tokenizer.decode(translated_text, skip_special_tokens=True)
    sub.text = translated_sentence

    pbar.update(1)
    return sub


def update_pbar(pbar, future):
    pbar.update(1)


def translate_subtitle_parallel(subtitle_file, output_file_path):
    print("Start translate_subtitle_parallel(" + subtitle_file + "," + output_file_path + ")")

    # Choose a model for Japanese to English translation
    model_name = "Helsinki-NLP/opus-mt-ja-en"

    # Download the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Read subtitle file
    subs = pysrt.open(subtitle_file)

    # Initialize concurrent futures executor
    num_workers = 4  # Number of parallel workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a tqdm progress bar to track the number of translated lines
        with tqdm(total=len(subs), unit="line", position=0, leave=True) as pbar:
            # Use ThreadPoolExecutor.map to process tasks in order
            list(executor.map(lambda sub: translate_line(sub, tokenizer, model, device, pbar), subs))

    # Save the filtered subtitles to a new SRT file
    subs.save(output_file_path, encoding='utf-8')


def create_empty_srt_file(file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("")


# Define the path to the video file
tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
sub_paths = filedialog.askopenfilenames()

for sub_path in sub_paths:
    print("Translating subtitle file for: " + sub_path)

    directory_path = os.path.dirname(sub_path)
    en_subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(sub_path))[0] + ".huggy.en.srt"
    create_empty_srt_file(en_subtitle_path)
    translate_subtitle_parallel(sub_path, en_subtitle_path)

    print("Translation complete. Translated SRT file:", en_subtitle_path)
