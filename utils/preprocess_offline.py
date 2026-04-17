import os
import torch
import numpy as np
import mne
from scipy import signal
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataset import build_global_summary_dict


def process_and_save_data():
    print("[*] Starting Offline Preprocessing with Overlap and NO Leakage...")

    for split in ['train', 'test']:
        for label in ['seizure', 'normal']:
            os.makedirs(os.path.join(config.PROCESSED_DIR, split, label), exist_ok=True)

    seizure_map = build_global_summary_dict(config.RAW_DIR)

    s_idx, n_idx = 0, 0
    window_samples = config.WINDOW_SIZE_SEC * config.FS
    step_samples = (config.WINDOW_SIZE_SEC - config.OVERLAP_SEC) * config.FS

    # נאסוף קודם כל את רשימת הקבצים שיש בהם התקפים כדי לפצל אותם
    all_seizure_files = []
    for subject in os.listdir(config.RAW_DIR):
        sub_path = os.path.join(config.RAW_DIR, subject)
        if not os.path.isdir(sub_path): continue
        for edf_file in os.listdir(sub_path):
            if edf_file.endswith('.edf') and edf_file in seizure_map and len(seizure_map[edf_file]) > 0:
                all_seizure_files.append((sub_path, edf_file))

    # חלוקה מראש: 80% מהקבצים לאימון, 20% לבחינה
    random.shuffle(all_seizure_files)
    split_idx = int(len(all_seizure_files) * 0.8)
    train_files = all_seizure_files[:split_idx]
    test_files = all_seizure_files[split_idx:]

    print(f"[*] Training on {len(train_files)} files, Testing on {len(test_files)} files.")

    # הפונקציה שתעבד ותשמור את הקבצים
    def process_file_list(file_list, folder):
        nonlocal s_idx, n_idx
        for sub_path, edf_file in file_list:
            seizures = seizure_map.get(edf_file, [])
            try:
                raw = mne.io.read_raw_edf(os.path.join(sub_path, edf_file), preload=True, verbose=False)
                raw.filter(0.5, 40.0, verbose=False)
                data = raw.get_data()

                # --- בדיקת הבטיחות החדשה ---
                # אם בקובץ יש פחות מ-23 ערוצים, המערכת תדלג עליו ותעבור לקובץ הבא
                if data.shape[0] < 23:
                    print(f"[!] Skipping {edf_file} - found only {data.shape[0]} channels.")
                    continue

                # אם יש 23 ומעלה, אנחנו חותכים את היתר כדי שנישאר בדיוק עם 23
                data = data[:23, :]
                x_s, x_n = [], []

                for i in range(0, data.shape[1] - window_samples, step_samples):
                    win = data[:, i: i + window_samples]
                    t_start = i / config.FS
                    t_end = t_start + config.WINDOW_SIZE_SEC

                    is_s = any(s_s <= t_end and s_e >= t_start for (s_s, s_e) in seizures)

                    f, t, Zxx = signal.stft(win, fs=config.FS, nperseg=256, noverlap=128)
                    stft_tensor = torch.tensor(np.abs(Zxx), dtype=torch.float32)

                    if is_s:
                        x_s.append(stft_tensor)
                    else:
                        x_n.append(stft_tensor)

                # שמירה לתיקייה שהוגדרה לקובץ (train או test)
                if x_s:
                    random.shuffle(x_n)
                    x_n = x_n[:len(x_s)]  # איזון

                    for t in x_s:
                        torch.save(t, os.path.join(config.PROCESSED_DIR, folder, "seizure", f"s_{s_idx}.pt"))
                        s_idx += 1
                    for t in x_n:
                        torch.save(t, os.path.join(config.PROCESSED_DIR, folder, "normal", f"n_{n_idx}.pt"))
                        n_idx += 1
                print(f"Finished {edf_file} -> {folder}")
            except Exception as e:
                print(f"[!] Failed to process {edf_file}: {e}")

    # הרצת העיבוד על הרשימות שיצרנו
    print("\n--- Processing Train Files ---")
    process_file_list(train_files, "train")
    print("\n--- Processing Test Files ---")
    process_file_list(test_files, "test")


if __name__ == "__main__":
    process_and_save_data()