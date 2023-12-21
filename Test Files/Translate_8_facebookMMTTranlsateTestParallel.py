import os
import tkinter
import concurrent.futures
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm
from tkinter import filedialog


def clean_up_line(line: str) -> str:
    line = line.replace("、", ", ")
    line = line.replace("・", " ")
    line = line.replace("…", "...")
    line = line.replace("。", ".")
    return line


def translate_line(line: str, tokenizer, model, device, pbar) -> str:
    encoded_text = tokenizer(line, return_tensors="pt", padding="max_length")
    # Run on GPU if available
    encoded_text = encoded_text.to(device)

    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    translated_sentence = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # print(f"Japanese text: {line}")
    # print(f"Translated text: {translated_sentence}")

    pbar.update(1)
    return translated_sentence


def update_pbar(pbar, future):
    pbar.update(1)


def translate_subtitle_parallel(subtitle_file, output_file_path):
    print("Start translate_subtitle_parallel(" + subtitle_file + "," + output_file_path + ")")

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    tokenizer.src_lang = "ja_XX"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Read subtitle file
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        subtitle_lines = file.readlines()

    # Get only the text lines, every 4th line starting at the 3rd line
    text_lines = subtitle_lines[2::4]

    batch_size = 10  # adjust based on your needs and memory constraints
    translated_lines = []
    with tqdm(total=len(text_lines), desc="Translating subtitles...") as pbar:
        for i in range(0, len(text_lines), batch_size):
            encoded_batch = tokenizer(text_lines[i:i + batch_size], return_tensors="pt", padding=True,
                                      truncation=True).to(device)
            generated_tokens = model.generate(**encoded_batch, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
            translated_lines.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
            pbar.update(batch_size)

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
sub_path = filedialog.askopenfilename()

print("Translating subtitle file for: " + sub_path)

directory_path = os.path.dirname(sub_path)
en_subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(sub_path))[0] + ".mmt.srt"
create_empty_srt_file(en_subtitle_path)
translate_subtitle_parallel(sub_path, en_subtitle_path)

print("Translation complete. Translated SRT file:", en_subtitle_path)
