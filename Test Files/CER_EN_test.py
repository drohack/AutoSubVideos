import pysrt
import jiwer
import re


subs_whisper = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.whisper.jp.huggy.en.srt")
subs_faster = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster.huggy.en.srt")
subs_faster_reduce = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce.huggy.en.srt")
subs_fr_medium = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-medium.huggy.en.srt")
subs_fr_medium_previous = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-medium-previous.huggy.en.srt")
subs_fr_t02 = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-t0.2.huggy.en.srt")
subs_fr_t05 = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-t0.5.huggy.en.srt")
subs_fr_q = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.stable_whisper.stable-jp-faster-reduce-q.huggy.en.srt")

subs_original = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01 - I'm Luffy! The Man Who's Gonna Be King of the Pirates! HDTV-1080p.huggy.en.srt")
subs_r = pysrt.open("C:\\Users\\droha\\Videos\\Subtitle AI\\CER_EN\\One Piece - S01E01.第1話　俺はルフィ！海賊王になる男だ！.WEBRip.Amazon.ja-jp[sdh].huggy.en.srt")

og = "\n".join([sub.text.lower() for sub in subs_original])
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

cer_og = jiwer.cer(truth=og, hypothesis=reference, truth_transform=transforms, hypothesis_transform=transforms)
cer_whisper = jiwer.cer(truth=reference, hypothesis=h_whisper, truth_transform=transforms, hypothesis_transform=transforms)
cer_faster = jiwer.cer(truth=reference, hypothesis=h_faster, truth_transform=transforms, hypothesis_transform=transforms)
cer_reduce = jiwer.cer(truth=reference, hypothesis=h_reduce, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_medium = jiwer.cer(truth=reference, hypothesis=h_fr_medium, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_medium_previous = jiwer.cer(truth=reference, hypothesis=h_fr_medium_previous, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_t02 = jiwer.cer(truth=reference, hypothesis=h_fr_t02, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_t05 = jiwer.cer(truth=reference, hypothesis=h_fr_t05, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_q = jiwer.cer(truth=reference, hypothesis=h_fr_q, truth_transform=transforms, hypothesis_transform=transforms)

print(f"Character Error Rate (CER) OG             :", cer_og)
print(f"Character Error Rate (CER) whisper        :", cer_whisper)
print(f"Character Error Rate (CER) faster         :", cer_faster)
print(f"Character Error Rate (CER) reduce         :", cer_reduce)
print(f"Character Error Rate (CER) medium         :", cer_fr_medium)
print(f"Character Error Rate (CER) medium_previous:", cer_fr_medium_previous)
print(f"Character Error Rate (CER) t0.2           :", cer_fr_t02)
print(f"Character Error Rate (CER) t0.5           :", cer_fr_t05)
print(f"Character Error Rate (CER) q              :", cer_fr_q)

cer_og = jiwer.compute_measures(truth=og, hypothesis=reference, truth_transform=transforms, hypothesis_transform=transforms)
cer_whisper = jiwer.compute_measures(truth=reference, hypothesis=h_whisper, truth_transform=transforms, hypothesis_transform=transforms)
cer_faster = jiwer.compute_measures(truth=reference, hypothesis=h_faster, truth_transform=transforms, hypothesis_transform=transforms)
cer_reduce = jiwer.compute_measures(truth=reference, hypothesis=h_reduce, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_medium = jiwer.compute_measures(truth=reference, hypothesis=h_fr_medium, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_medium_previous = jiwer.compute_measures(truth=reference, hypothesis=h_fr_medium_previous, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_t02 = jiwer.compute_measures(truth=reference, hypothesis=h_fr_t02, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_t05 = jiwer.compute_measures(truth=reference, hypothesis=h_fr_t05, truth_transform=transforms, hypothesis_transform=transforms)
cer_fr_q = jiwer.compute_measures(truth=reference, hypothesis=h_fr_q, truth_transform=transforms, hypothesis_transform=transforms)

print(f"jiwer OG             :", cer_og)
print(f"jiwer whisper        :", cer_whisper)
print(f"jiwer faster         :", cer_faster)
print(f"jiwer reduce         :", cer_reduce)
print(f"jiwer medium         :", cer_fr_medium)
print(f"jiwer medium_previous:", cer_fr_medium_previous)
print(f"jiwer t0.2           :", cer_fr_t02)
print(f"jiwer t0.5           :", cer_fr_t05)
print(f"jiwer q              :", cer_fr_q)
