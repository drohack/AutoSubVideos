import pysrt
import jiwer
import re


subs_whisper = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_JP\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.whisper.jp.srt")
subs_faster = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_JP\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster.srt")
subs_faster_reduce = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_JP\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce.srt")
subs_fr_medium = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_JP\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-medium.srt")
subs_fr_medium_previous = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_JP\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-medium-previous.srt")
subs_fr_t02 = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_JP\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-t0.2.srt")
subs_fr_t05 = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_JP\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-t0.5.srt")
subs_fr_q = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_JP\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-q.srt")

subs_r = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_JP\\One Piece - S01E01.第1話　俺はルフィ！海賊王になる男だ！.WEBRip.Amazon.ja-jp[sdh].srt")

h_whisper = "\n".join([sub.text.lower() for sub in subs_whisper])  # Lowercase for consistency
h_faster = "\n".join([sub.text.lower() for sub in subs_faster])  # Lowercase for consistency
h_reduce = "\n".join([sub.text.lower() for sub in subs_faster_reduce])  # Lowercase for consistency
h_fr_medium = "\n".join([sub.text.lower() for sub in subs_fr_medium])  # Lowercase for consistency
h_fr_medium_previous = "\n".join([sub.text.lower() for sub in subs_fr_medium_previous])  # Lowercase for consistency
h_fr_t02 = "\n".join([sub.text.lower() for sub in subs_fr_t02])  # Lowercase for consistency
h_fr_t05 = "\n".join([sub.text.lower() for sub in subs_fr_t05])  # Lowercase for consistency
h_fr_q = "\n".join([sub.text.lower() for sub in subs_fr_q])  # Lowercase for consistency
reference = "\n".join([
    re.sub(r"\（.*?\）", "", sub.text).strip()  # Remove text within parentheses
    for sub in subs_r
    if not re.match(r"^\（.*?\）$", sub.text)  # Exclude lines with only parentheses
])

cer_whisper = jiwer.cer(truth=reference, hypothesis=h_whisper)
cer_faster = jiwer.cer(truth=reference, hypothesis=h_faster)
cer_reduce = jiwer.cer(truth=reference, hypothesis=h_reduce)
cer_fr_medium = jiwer.cer(truth=reference, hypothesis=h_fr_medium)
cer_fr_medium_previous = jiwer.cer(truth=reference, hypothesis=h_fr_medium_previous)
cer_fr_t02 = jiwer.cer(truth=reference, hypothesis=h_fr_t02)
cer_fr_t05 = jiwer.cer(truth=reference, hypothesis=h_fr_t05)
cer_fr_q = jiwer.cer(truth=reference, hypothesis=h_fr_q)

print(f"Character Error Rate (CER) whisper        :", cer_whisper)
print(f"Character Error Rate (CER) faster         :", cer_faster)
print(f"Character Error Rate (CER) reduce         :", cer_reduce)
print(f"Character Error Rate (CER) medium         :", cer_fr_medium)
print(f"Character Error Rate (CER) medium_previous:", cer_fr_medium_previous)
print(f"Character Error Rate (CER) t0.2           :", cer_fr_t02)
print(f"Character Error Rate (CER) t0.5           :", cer_fr_t05)
print(f"Character Error Rate (CER) q              :", cer_fr_q)
