# Built to .exe
`pyinstaller --onefile --hidden-import=pytorch --collect-data torch --recursive-copy-metadata torch --recursive-copy-metadata tqdm --recursive-copy-metadata sacremoses --recursive-copy-metadata tokenizers --recursive-copy-metadata importlib_metadata --recursive-copy-metadata transformers --recursive-copy-metadata sentencepiece --recursive-copy-metadata regex main.py`
