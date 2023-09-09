import os
import re
import sys

def preprocess(text):
    # Separate punctuation from words
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    # Lowercase all words
    text = text.lower()
    return text

def preprocess_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)

        with open(input_file, "r", encoding='utf-8') as f:
            text = f.read()

        preprocessed_text = preprocess(text)

        with open(output_file, "w", encoding='utf-8') as f:
            f.write(preprocessed_text)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)
    preprocess_files(input_dir, output_dir)
