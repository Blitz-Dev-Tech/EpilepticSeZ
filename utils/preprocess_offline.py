import os
import torch
import numpy as np
import mne
from scipy import signal
import random
import sys

# הוספת התיקייה הראשית לנתיב
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataset import build_global_summary_dict


def process_and_save_data():
    print("[*] Starting Offline Preprocessing with Overlap...")

    # יצירת מבנה תיקיות
    for split in ['train', 'test']:
        for label in ['seizure', 'normal']:
            os.makedirs(os.path.join(config.PROCESSED_DIR, split, label), exist_ok=True)

    seizure_map = build_global_summary_dict(config.RAW_DIR)

    s_idx, n_idx = 0, 0
    window_samples = config.WINDOW_SIZE_SEC * config.FS
    step_samples = (config.WINDOW_SIZE_SEC - config.OVERLAP_SEC) * config.FS

    for subject in os.listdir(config.RAW_DIR):
        sub_path = os.path.join(config.RAW_DIR, subject)
        if not os.path.isdir(sub_path): continue

        for edf_file in os.listdir(sub_path):
            if not edf_file.endswith('.edf'): continue

            seizures = seizure_map.get(edf_file, [])
            raw = mne.io.read_raw_edf(os.path.join(sub_path, edf_file), preload=True, verbose=False)
            raw.filter(0.5, 40.0, verbose=False)
            data = raw.get_data()

            x_s, x_n = [], []

            # לולאה עם קפיצות של step_samples (מייצר חפיפה)
            for i in range(0, data.shape[1] - window_samples, step_samples):
                win = data[:, i: i + window_samples]
                t_start = i / config.FS
                t_end = t_start + config.WINDOW_SIZE_SEC

                is_s = any(s_s <= t_end and s_e >= t_start for (s_s, s_e) in seizures)

                # חישוב STFT
                f, t, Zxx = signal.stft(win, fs=config.FS, nperseg=256, noverlap=128)
                stft_tensor = torch.tensor(np.abs(Zxx), dtype=torch.float32)

                if is_s:
                    x_s.append(stft_tensor)
                else:
                    x_n.append(stft_tensor)

            # איזון ושמירה
            if x_s:
                random.shuffle(x_n)
                x_n = x_n[:len(x_s)]

                for t in x_s:
                    folder = "train" if random.random() < 0.8 else "test"
                    torch.save(t, os.path.join(config.PROCESSED_DIR, folder, "seizure", f"s_{s_idx}.pt"))
                    s_idx += 1
                for t in x_n:
                    folder = "train" if random.random() < 0.8 else "test"
                    torch.save(t, os.path.join(config.PROCESSED_DIR, folder, "normal", f"n_{n_idx}.pt"))
                    n_idx += 1
            print(f"Finished {edf_file}")


if __name__ == "__main__":
    process_and_save_data()
