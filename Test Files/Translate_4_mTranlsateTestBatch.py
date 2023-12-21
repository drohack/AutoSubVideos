import os
import tkinter
import concurrent.futures
import json
import math
from tqdm import tqdm
import datetime
import openai
from mtranslate import translate
from googletrans import Translator
from tkinter import filedialog

OPEN_AI_API = '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'
openai.api_key = OPEN_AI_API


def print_with_timestamp(message):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(formatted_time, message)


def translate_text(string_to_translate: str):
    #print_with_timestamp("Start translate_subtitle(" + string_to_translate + ")")
    #translated_text = translate(string_to_translate, from_language='ja', to_language='en')
    translator = Translator()
    translated_text = translator.translate(string_to_translate, src='ja', dest='en').text
    return translated_text


def openai_translate(string_to_translate: str):
    prompt = "Translate the following JSON object from Japanese to English: " + string_to_translate
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=256  # Adjust as needed
    )
    return response.choices[0].text


def create_empty_srt_file(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("")


# Define the path to the video file
tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
sub_path = filedialog.askopenfilename()

print_with_timestamp("Translating subtitle file for: " + sub_path)

# Create a numbered array for each line in the file
numbered_lines = {}

with open(sub_path, 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file, start=1):
        numbered_lines[line_number] = line.strip()

# Then grab only the lines we want to translate
lines_to_translate = {}
for key, value in numbered_lines.items():
    if '-->' not in value and not value.isdigit() and value != "":
        lines_to_translate[key] = value

# Loop through the lines_to_translate and remove them from the numbered_lines
for key, value in lines_to_translate.items():
    if key in numbered_lines:
        del numbered_lines[key]

# Loop through all of the lines_to_translate and split it up into 4,000 character batches
# First get the number of items to add to the array by the character count devided by 4000
#char_count = len(json.dumps(lines_to_translate, ensure_ascii=False))
#num_items = math.ceil(char_count / 4000)
# Next create an array and populate with the lines
lines_to_translate_array = []
array_index = 0
'''
for key, value in lines_to_translate.items():
    if not lines_to_translate_array:
        lines_to_translate_array.append([{key: value}])
    elif (len(json.dumps(lines_to_translate_array[array_index], ensure_ascii=False)) + len(json.dumps({key: value}, ensure_ascii=False))) > 4000:
        lines_to_translate_array.append([{key: value}])
        array_index += 1
    else:
        lines_to_translate_array[array_index].append({key: value})
'''
for key, value in lines_to_translate.items():
    if not lines_to_translate_array:
        lines_to_translate_array.append([value])
    elif (len(json.dumps(lines_to_translate_array[array_index], ensure_ascii=False)) + len(value)) > 4000:
        lines_to_translate_array.append([value])
        array_index += 1
    else:
        lines_to_translate_array[array_index].append(value)

'''
print("char_count:" + str(char_count))
print("num_items:" + str(num_items))
print("lines_to_translate_array size:" + str(len(lines_to_translate_array)))
for a in lines_to_translate_array:
    print("lines_to_translate_array size:" + str(len(a)))
    print("lines_to_translate_array size:" + str(a))
'''

# Loop over all of the lines to translate and translate them
print_with_timestamp("Start translate lines")
'''
translated_lines_array = [None] * len(lines_to_translate_array)
for i in range(len(lines_to_translate_array)):
    # Convert lines_to_translate to a json string
    line_to_translate_str = json.dumps(lines_to_translate_array[i], ensure_ascii=False)
    #print_with_timestamp(line_to_translate_str)
    translated_lines_array[i] = translate_text(line_to_translate_str)
print_with_timestamp(translated_lines_array[0])
'''
line_to_translate = json.dumps(lines_to_translate_array[0], ensure_ascii=False)
#line_to_translate = line_to_translate[1:-1]
line_to_translate = line_to_translate.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("\"", "")
print_with_timestamp(line_to_translate)
translated_line = translate_text(line_to_translate)
print_with_timestamp(translated_line)

print_with_timestamp("End translate lines")

'''
directory_path = os.path.dirname(sub_path)
en_subtitle_path = directory_path + "/" + os.path.splitext(os.path.basename(sub_path))[0] + ".en.srt"
create_empty_srt_file(en_subtitle_path)
'''

print_with_timestamp("Translation complete. Translated SRT file: ")
