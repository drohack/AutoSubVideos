import os
import pysrt
import jiwer
import re


folder_path = 'C:\\Users\\droha\\Videos\\Subtitle AI\\One Piece\\CER_JP'

subs_r = pysrt.open(folder_path + "\\One Piece - S01E01.第1話　俺はルフィ！海賊王になる男だ！.WEBRip.Amazon.ja-jp[sdh].srt")
reference = "\n".join([
    re.sub(r"\（.*?\）", "", re.sub(r'[—～…･”“　]', '', re.sub('！', '!', re.sub('？', '?', sub.text)))).strip()  # Remove text within parentheses
    for sub in subs_r
    if not re.match(r"^\（.*?\）$", sub.text)  # Exclude lines with only parentheses
])

# List all files in the folder
file_list = os.listdir(folder_path)

# Loop through the files
for file_name in file_list:
    # Optionally, you can check if the item is a file (not a directory) before processing
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and "One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p" in file_name:
        f_name = re.sub("One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.", '', file_name)
        f_name = re.sub('.srt', '', f_name)
        # Add your file processing logic here
        subs = pysrt.open(file_path)
        lines = "\n".join([sub.text for sub in subs])  # Lowercase for consistency
        cer = jiwer.cer(truth=reference, hypothesis=lines)
        print(f"Character Error Rate (CER) {f_name}:", cer)
