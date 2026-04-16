# utils/dataset.py
import os
import re
import mne
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
import sys

# הוספת התיקייה הראשית לנתיב כדי לייבא את config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_global_summary_dict(root_dir):
    """
    סורק את כל תיקיות המטופלים (chb01, chb02...) ובונה מפה של כל ההתקפים.
    מחזיר מילון: { 'שם_קובץ.edf': [(התחלה1, סוף1), (התחלה2, סוף2)] }
    """
    global_summary = {}

    # מעבר על כל התיקיות ב-data/raw/
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path): continue

        # חיפוש קובץ ה-summary בתוך התיקייה
        summary_files = [f for f in os.listdir(subdir_path) if f.endswith('-summary.txt')]
        if not summary_files: continue

        summary_path = os.path.join(subdir_path, summary_files[0])

        try:
            with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # פיצול לפי בלוקים של קבצים
            file_blocks = content.split('File Name:')
            for block in file_blocks[1:]:
                # חילוץ שם הקובץ
                file_name_match = re.search(r'(\w+\d+_\d+\+?\.edf)', block)
                if not file_name_match: continue
                file_name = file_name_match.group(1)

                # חילוץ זמני התחלה וסיום
                starts = re.findall(r'Seizure \d* ?Start Time:\s*(\d+)', block)
                ends = re.findall(r'Seizure \d* ?End Time:\s*(\d+)', block)

                seizures = [(int(s), int(e)) for s, e in zip(starts, ends)]
                global_summary[file_name] = seizures
        except Exception as e:
            print(f"[!] Error parsing summary for {subdir}: {e}")

    return global_summary


class CHBMITDataset(Dataset):
    def __init__(self, edf_path, seizure_times, window_size_sec=4):
        """
        edf_path: נתיב מלא לקובץ ה-EDF
        seizure_times: רשימת טאפלים של (התחלה, סוף) בשניות
        window_size_sec: גודל חלון הזמן (דיפולט 4 שניות)
        """
        self.edf_path = edf_path
        self.window_size_sec = window_size_sec
        self.seizure_times = seizure_times

        # טעינת המידע (רק כותרות בשלב זה כדי לחסוך זיכרון)
        try:
            self.raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            self.fs = int(self.raw.info['sfreq'])  # בדרך כלל 256
            self.data = self.raw.get_data()

            # נרמול פשוט (Zero Mean, Unit Variance)
            self.data = (self.data - np.mean(self.data)) / (np.std(self.data) + 1e-6)

            self.window_samples = window_size_sec * self.fs
            self.num_windows = self.data.shape[1] // self.window_samples
        except Exception as e:
            print(f"[!] Error loading {edf_path}: {e}")
            self.num_windows = 0

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        # 1. חילוץ חלון הזמן הגולמי
        start_sample = idx * self.window_samples
        end_sample = start_sample + self.window_samples
        x_raw = self.data[:, start_sample:end_sample]

        # 2. המרה ל-STFT (הפיכה לתמונת תדר/זמן)
        # nperseg=256 אומר שאנחנו מנתחים כל שנייה בנפרד בתוך ה-4 שניות
        f, t, Zxx = signal.stft(x_raw, fs=self.fs, nperseg=256, noverlap=128)

        # לוקחים ערך מוחלט (Magnitude) כי ה-CNN לא עובד עם מספרים מרוכבים
        x_stft = np.abs(Zxx)

        # המרה לטנזור של PyTorch במבנה [Channels, Freq, Time]
        x_tensor = torch.from_numpy(x_stft).float()

        # 3. קביעת התווית (Label)
        # מחשבים את זמן החלון בשניות
        win_start_sec = start_sample / self.fs
        win_end_sec = end_sample / self.fs

        label = 0
        # בדיקה אם החלון נופל בתוך אחד מזמני ההתקפים
        for s_start, s_end in self.seizure_times:
            if (win_end_sec > s_start) and (win_start_sec < s_end):
                label = 1
                break

        return x_tensor, torch.tensor(label, dtype=torch.long)