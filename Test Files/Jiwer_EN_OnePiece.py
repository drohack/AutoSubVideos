import os
import pysrt
import jiwer
import re


transforms = jiwer.Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

folder_path = 'C:\\Users\\droha\\Videos\\Subtitle AI\\One Piece\\CER_EN'

subs_original = pysrt.open(folder_path + "\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.huggy.en.srt")
subs_r = pysrt.open(folder_path + "\\One Piece - S01E01.第1話　俺はルフィ！海賊王になる男だ！.WEBRip.Amazon.ja-jp[sdh].huggy.en.srt")
og = "\n".join([sub.text for sub in subs_original])
reference = "\n".join([
    re.sub(r"\（.*?\）", "", re.sub(r'[—～…･”“　]', '', re.sub('！', '!', re.sub('？', '?', sub.text)))).strip()  # Remove text within parentheses
    for sub in subs_r
    if not re.match(r"^\（.*?\）$", sub.text)  # Exclude lines with only parentheses
])
wer_og = jiwer.wer(truth=og, hypothesis=reference, truth_transform=transforms, hypothesis_transform=transforms)
print(f"Word Error Rate (WER) OG:", wer_og)

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
        wer = jiwer.wer(truth=reference, hypothesis=lines, truth_transform=transforms, hypothesis_transform=transforms)
        print(f"Word Error Rate (WER) {f_name}:", wer)
