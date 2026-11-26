"""
Script to download Wikipedia pages from the WikiQA dataset.
Includes retry logic and manually inserted titles that needed special handling.
"""

import os
from tqdm import tqdm
import wikipedia
from datasets import load_dataset
from wikipedia.exceptions import DisambiguationError, PageError

# --------------------------------------------
# Step 1: Load the dataset and collect titles
# --------------------------------------------

def get_unique_titles(dataset):
    """
    Extract unique document titles with positive labels from WikiQA dataset.
    """
    titles = []
    for example in dataset:
        if example["document_title"] not in titles and example['label'] == 1:
            titles.append(example["document_title"])
    return titles

# Load WikiQA dataset
dataset = load_dataset("microsoft/wiki_qa")['train']
titles = get_unique_titles(dataset)

# --------------------------------------------
# Step 2: Save Wikipedia articles
# --------------------------------------------

def save_wikipedia_pages(title_list, output_dir="input/documents", fail_log="failed_titles.txt"):
    """
    Attempt to download and save Wikipedia articles for a list of titles.
    Saves successful ones as .txt files and logs failures.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(fail_log, "w", encoding="utf-8") as failed_file:
        for title in tqdm(title_list, desc="Saving Wikipedia titles"):
            try:
                article = wikipedia.page(title, auto_suggest=False)
                filename = os.path.join(output_dir, f"{title.replace('/', '_')}.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(article.content)
            except Exception as e:
                failed_file.write(f"{title}\t{str(e)}\n")

# Uncomment to run initial save
# save_wikipedia_pages(titles)

# --------------------------------------------
# Step 3: Retry failed titles (if needed)
# --------------------------------------------

def retry_failed_titles(failed_file="failed_titles.txt", output_dir="input/documents", detailed_log="detailed_failed_log.txt"):
    """
    Retry downloading titles that previously failed, and log the detailed errors.
    """
    with open(failed_file, "r", encoding="utf-8") as f:
        failed_titles = [line.split('\t')[0] for line in f if line.strip()]

    with open(detailed_log, "w", encoding="utf-8") as log:
        for title in tqdm(failed_titles, desc="Re-checking failed titles"):
            try:
                article = wikipedia.page(title, auto_suggest=False)
                filename = os.path.join(output_dir, f"{title.replace('/', '_')}.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(article.content)
            except Exception as e:
                log.write(f"{title}\t{str(e)}\n")

# --------------------------------------------
# Step 4: Manual insert of tricky titles
# --------------------------------------------

# These were added manually because they were either missed or caused issues (e.g., disambiguation).
# Includes "Glow-in-the-dark" and "Add-on"
manual_titles = [
    "Joint committee (legislative)", "The Estée Lauder Companies", "Judas (song)",
    "Heroes (American TV series)", "Feelin' Alright", "Stand by Me (Ben E. King song)",
    "Mary Poppins (film)", "Mary Poppins (book series)", "What It Takes (Aerosmith song)",
    "Microsoft Bing", "Enigma (German band)", "Glow-in-the-dark", "Full-time job",
    "Add-on", "World Trade Center (1973-2001)"
]

def handle_manual_titles(titles, output_dir="input/documents", fail_log="new_failed_titles.txt"):
    """
    Handle tricky titles, including disambiguation pages.
    """
    with open(fail_log, "w", encoding="utf-8") as failed_file:
        for title in tqdm(titles, desc='Manual download of tricky pages'):
            try:
                article = wikipedia.page(title, auto_suggest=False)
            except DisambiguationError as e:
                try:
                    article = wikipedia.page(title=e.title, auto_suggest=False, preload=False)
                except Exception as e2:
                    failed_file.write(f"{title}\tDisambiguationError Retry Failed: {str(e2)}\n")
                    continue
            except PageError as e:
                failed_file.write(f"{title}\tPageError: {str(e)}\n")
                continue
            except Exception as e:
                failed_file.write(f"{title}\tUnexpected Error: {str(e)}\n")
                continue

            filename = os.path.join(output_dir, f"{title.replace('/', '_')}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(article.content)

# Uncomment to run manually
# handle_manual_titles(manual_titles)

# --------------------------------------------
# Step 5: Validate which titles were downloaded
# --------------------------------------------

def check_downloaded_titles(titles, folder="./input/documents"):
    """
    Check which titles were successfully downloaded and which are missing.
    """
    downloaded = []
    missing = []

    for title in titles:
        filename = os.path.join(folder, f"{title.replace('/', '_')}.txt")
        if os.path.isfile(filename):
            downloaded.append(title)
        else:
            missing.append(title)

    print(f"\nDownloaded ({len(downloaded)}):")
    # for title in downloaded:
    #     print(f"✓ {title}")

    print(f"\nMissing ({len(missing)}):")
    for title in missing:
        print(f"✗ {title}")

# Uncomment to check
# check_downloaded_titles(titles + manual_titles)
