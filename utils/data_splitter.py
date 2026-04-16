# utils/data_splitter.py
import os
import json
import random
import sys

# הוספת התיקייה הראשית לנתיב כדי שנוכל לייבא את config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_all_edf_files(data_dir):
    """סורק את כל תתי-התיקיות ומחזיר רשימה של כל קבצי ה-EDF"""
    edf_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.edf'):
                # שומר את הנתיב היחסי, למשל: chb01/chb01_03.edf
                rel_path = os.path.relpath(os.path.join(root, file), data_dir)
                # המרת לוכסנים לאחור (בווינדוס) ללוכסנים רגילים
                rel_path = rel_path.replace('\\', '/')
                edf_files.append(rel_path)
    return edf_files


def get_seizure_files_from_records(records_file_path):
    """קורא את קובץ ה-RECORDS-WITH-SEIZURES.txt ומחזיר רשימת קבצים עם התקפים"""
    seizure_files = []
    if not os.path.exists(records_file_path):
        print(f"[!] Warning: Records file not found at {records_file_path}")
        return seizure_files

    with open(records_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # מחליף לוכסנים כדי למנוע בעיות בין ווינדוס ללינוקס
                seizure_files.append(line.replace('\\', '/'))
    return seizure_files


def split_data(train_ratio=0.8, seed=42):
    """מבצע את החלוקה המרובדת שומרת את התוצאות ל-JSON"""
    random.seed(seed)

    print("[*] Starting data split process...")
    all_files = get_all_edf_files(config.DATA_DIR)
    known_seizure_files = get_seizure_files_from_records(config.RECORDS_FILE)

    print(f"[*] Found {len(all_files)} total EDF files in data directory.")
    print(f"[*] Found {len(known_seizure_files)} files marked with seizures in records.")

    # סינון: רק קבצים שבאמת קיימים אצלנו בתיקייה (למקרה שלא הורדת את כל המטופלים עדיין)
    existing_seizure_files = [f for f in all_files if f in known_seizure_files]
    normal_files = [f for f in all_files if f not in known_seizure_files]

    # ערבוב אקראי
    random.shuffle(existing_seizure_files)
    random.shuffle(normal_files)

    # חיתוך
    seizure_split = int(len(existing_seizure_files) * train_ratio)
    normal_split = int(len(normal_files) * train_ratio)

    train_files = existing_seizure_files[:seizure_split] + normal_files[:normal_split]
    test_files = existing_seizure_files[seizure_split:] + normal_files[normal_split:]

    # ערבוב סופי של כל קבוצה
    random.shuffle(train_files)
    random.shuffle(test_files)

    # שמירה לדיסק
    train_json = os.path.join(config.SPLITS_DIR, "train_files.json")
    test_json = os.path.join(config.SPLITS_DIR, "test_files.json")

    with open(train_json, 'w') as f:
        json.dump(train_files, f, indent=4)

    with open(test_json, 'w') as f:
        json.dump(test_files, f, indent=4)

    print(f"[V] Successfully saved {len(train_files)} train files to {train_json}")
    print(f"[V] Successfully saved {len(test_files)} test files to {test_json}")


if __name__ == "__main__":
    split_data()
