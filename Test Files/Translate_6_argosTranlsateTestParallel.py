import os
import tkinter
import concurrent.futures
import argostranslate.package
import argostranslate.translate
from tqdm import tqdm
from tkinter import filedialog


def clean_up_line(line: str) -> str:
    line = line.replace("、", ", ")
    line = line.replace("・", " ")
    line = line.replace("…", "...")
    line = line.replace("。", ".")
    return line


def translate_line(line: str, from_code: str, to_code: str, pbar) -> str:
    translated_text = argostranslate.translate.translate(q=clean_up_line(line), from_code=from_code, to_code=to_code)

    pbar.update(1)
    return translated_text


def update_pbar(pbar, future):
    pbar.update(1)


def translate_subtitle_parallel(subtitle_file, output_file_path):
    print("Start translate_subtitle_parallel(" + subtitle_file + "," + output_file_path + ")")

    from_code = "ja"
    to_code = "en"

    # Download and install Argos Translate package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

    # Read subtitle file
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        subtitle_lines = file.readlines()

    # Get only the text lines, every 4th line starting at the 3rd line
    text_lines = subtitle_lines[2::4]

    # Initialize concurrent futures executor
    num_workers = 4  # Number of parallel workers
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
        with tqdm(total=len(text_lines), unit="line", position=0, leave=True) as pbar:
            # Use ThreadPoolExecutor.map to process tasks in order
            translated_lines = list(executor.map(lambda line: translate_line(line, from_code, to_code, pbar), text_lines))

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
en_subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(sub_path))[0] + ".argos.srt"
create_empty_srt_file(en_subtitle_path)
translate_subtitle_parallel(sub_path, en_subtitle_path)

print("Translation complete. Translated SRT file:", en_subtitle_path)
