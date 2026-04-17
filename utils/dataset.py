import os
import re
import torch
from torch.utils.data import Dataset

def build_global_summary_dict(root_dir):
    """
    סורק את כל תיקיות המטופלים ובונה מפה של כל ההתקפים.
    דרוש עבור שלב ה-Preprocessing האופליין.
    """
    global_summary = {}

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path): continue

        summary_files = [f for f in os.listdir(subdir_path) if f.endswith('-summary.txt')]
        if not summary_files: continue

        summary_path = os.path.join(subdir_path, summary_files[0])

        try:
            with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            file_blocks = content.split('File Name:')
            for block in file_blocks[1:]:
                file_name_match = re.search(r'(\w+\d+_\d+\+?\.edf)', block)
                if not file_name_match: continue
                file_name = file_name_match.group(1)

                starts = re.findall(r'Seizure \d* ?Start Time:\s*(\d+)', block)
                ends = re.findall(r'Seizure \d* ?End Time:\s*(\d+)', block)

                seizures = [(int(s), int(e)) for s, e in zip(starts, ends)]
                global_summary[file_name] = seizures
        except Exception as e:
            print(f"[!] Error parsing summary for {subdir}: {e}")

    return global_summary

class EEGProcessedDataset(Dataset):
    def __init__(self, data_dir):
        """
        טעינת חלונות STFT מוכנים (קבצי .pt)
        """
        self.file_paths = []
        self.labels = []

        for label_name, label_val in [('seizure', 1), ('normal', 0)]:
            category_dir = os.path.join(data_dir, label_name)
            if os.path.exists(category_dir):
                for f in os.listdir(category_dir):
                    if f.endswith('.pt'):
                        self.file_paths.append(os.path.join(category_dir, f))
                        self.labels.append(label_val)

        print(f"Loaded {len(self.file_paths)} windows from {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = torch.load(self.file_paths[idx], weights_only=True)
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y