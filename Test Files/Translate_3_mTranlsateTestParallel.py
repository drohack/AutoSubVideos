import os
import tkinter
import concurrent.futures
from tqdm import tqdm
from mtranslate import translate
from tkinter import filedialog


def translate_line(line):
    # Translate a single line of text
    if '-->' in line or line.isdigit() or line == "":
        translated_text = line  # Keep the timestamp line unchanged
    else:
        translated_text = translate(line, 'en')
    return translated_text


def update_pbar(pbar, future):
    pbar.update(1)


def translate_subtitle_parallel(subtitle_file, output_file_path):
    print("Start translate_subtitle_parallel(" + subtitle_file + "," + output_file_path + ")")

    # Read subtitle file
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        subtitle_lines = file.readlines()

    # Initialize concurrent futures executor
    num_workers = 100  # Number of parallel workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        translated_lines = []

        '''
        # Create a tqdm progress bar to track the number of translated lines
        with tqdm(total=len(subtitle_lines), unit="line", position=0, leave=True) as pbar:
            # Submit tasks to the executor
            futures = []
            for line in subtitle_lines:
                future = executor.submit(translate_line, line)
                future.add_done_callback(lambda f: update_pbar(pbar, f))
                futures.append(future)

            # Retrieve results from futures
            for future in concurrent.futures.as_completed(futures):
                translated_lines.append(future.result())
        '''

        # Create a tqdm progress bar to track the number of translated lines
        with tqdm(total=len(subtitle_lines), unit="line", position=0, leave=True) as pbar:
            # Use ThreadPoolExecutor.map to process tasks in order
            translated_lines = list(executor.map(translate_line, subtitle_lines))
            pbar.update(len(subtitle_lines))

    # Write translated lines to the output SRT file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + '\n')


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
translate_subtitle_parallel(sub_path, en_subtitle_path)

print("Translation complete. Translated SRT file:", en_subtitle_path)
