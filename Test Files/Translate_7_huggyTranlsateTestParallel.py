import os
import tkinter
import concurrent.futures
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from tkinter import filedialog


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
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        subtitle_lines = file.readlines()

    # Get only the text lines, every 4th line starting at the 3rd line
    text_lines = subtitle_lines[2::4]

    for line in text_lines:
        line = clean_up_line(line)

    # Initialize concurrent futures executor
    num_workers = 4  # Number of parallel workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        translated_lines = []

        # Create a tqdm progress bar to track the number of translated lines
        with tqdm(total=len(text_lines), unit="line", position=0, leave=True) as pbar:
            # Use ThreadPoolExecutor.map to process tasks in order
            translated_lines = list(executor.map(lambda line: translate_line(line, tokenizer, model, device, pbar), text_lines))

    # Replace the japanese lines with the translated lines (this keeps the index, and timestamps)
    subtitle_lines[2::4] = translated_lines

    # Write translated lines to the output SRT file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for line in subtitle_lines:
            if line.endswith('\n'):
                f.write(line)
            else:
                f.write(line + '\n')


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
