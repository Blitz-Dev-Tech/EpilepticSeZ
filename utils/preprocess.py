import os
import mne
import numpy as np
from eeg_parser import parse_summary

# --- הגדרות כלליות קבועות ---
SFREQ = 256
WINDOW_SIZE_SEC = 4
WINDOW_SIZE = WINDOW_SIZE_SEC * SFREQ


def create_segments(edf_path, seizure_times):
    """
    קורא קובץ EDF, חותך לחלונות ומאזן את הנתונים.
    מחזיר X (נתונים) ו-y (תגיות).
    """
    # טעינה וסינון רעשים חשמליים (verbose=False משתיק את ההדפסות המיותרות של הספריה)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    data = raw.get_data()  # הצורה של המטריצה: (ערוצים, כמות דגימות בזמן)

    x_seizure = []
    x_normal = []

    # ריצה על האות בחלונות של 4 שניות
    for i in range(0, data.shape[1] - WINDOW_SIZE, WINDOW_SIZE):
        window = data[:, i: i + WINDOW_SIZE]
        start_sec = i / SFREQ
        end_sec = start_sec + WINDOW_SIZE_SEC

        # בדיקה האם החלון הנוכחי נופל בתוך זמן של התקף
        is_seizure = False
        for (s_start, s_end) in seizure_times:
            # אם יש חפיפה כלשהי בין החלון שלנו לזמן ההתקף הרשום
            if start_sec <= s_end and end_sec >= s_start:
                is_seizure = True
                break

        if is_seizure:
            x_seizure.append(window)
        else:
            x_normal.append(window)

    # אם הקובץ לא מכיל שום התקף, אנחנו כרגע מדלגים עליו כדי לייצר דאטא-סט קומפקטי וממוקד לשלב ה-POC
    if len(x_seizure) == 0:
        return np.array([]), np.array([])

    # איזון הדאטא: מערבבים את חלונות הרקע ולוקחים כמות ששווה בדיוק לכמות ההתקפים
    np.random.shuffle(x_normal)
    x_normal_balanced = x_normal[:len(x_seizure)]

    # חיבור הסגמנטים של ההתקפים ושל המצב הרגיל למערך אחד
    X = np.concatenate([x_seizure, x_normal_balanced])

    # יצירת התגיות: 1 עבור התקף, 0 עבור מצב רגיל
    y = np.concatenate([np.ones(len(x_seizure)), np.zeros(len(x_normal_balanced))])

    return X, y


def process_all_data(raw_data_dir, processed_data_dir):
    """
    עובר על כל התיקיות ב-raw, חולץ את הנתונים הרלוונטיים ושומר ב-processed.
    """
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    # מעבר על כל תיקיות הנבדקים (chb01, chb02...)
    for subject_folder in os.listdir(raw_data_dir):
        sub_path = os.path.join(raw_data_dir, subject_folder)

        if os.path.isdir(sub_path) and subject_folder.startswith('chb'):
            print(f"\n--- Processing subject: {subject_folder} ---")

            # הדפסת הדיבאג: מה פייתון באמת רואה בתוך התיקייה?
            files_in_folder = os.listdir(sub_path)
            print(f"DEBUG: Files actually found in folder: {files_in_folder}")

            # מציאת קובץ הסיכום של הנבדק
            summary_file = os.path.join(sub_path, f"{subject_folder}-summary.txt")
            print(f"DEBUG: Code is looking EXACTLY for this path: {summary_file}")

            # ניסיון חלופי למקרה של סיומת כפולה (chb01-summary.txt.txt)
            fallback_summary = os.path.join(sub_path, f"{subject_folder}-summary.txt.txt")

            if os.path.exists(summary_file):
                print("DEBUG: Found summary file perfectly!")
                active_summary = summary_file
            elif os.path.exists(fallback_summary):
                print("DEBUG: Found summary file with double .txt extension! Using it.")
                active_summary = fallback_summary
            else:
                print(f"ERROR: Summary file STILL not found. Skipping {subject_folder}.")
                continue

            seizure_map = parse_summary(active_summary)

            subject_X = []
            subject_y = []

            # מעבר על כל קבצי ה-EDF של הנבדק הנוכחי
            for edf_file in os.listdir(sub_path):
                if edf_file.endswith('.edf'):
                    edf_path = os.path.join(sub_path, edf_file)
                    seizures = seizure_map.get(edf_file, [])

                    if len(seizures) > 0:
                        print(f"  - Extracting windows from {edf_file}...")
                        X, y = create_segments(edf_path, seizures)
                        if len(X) > 0:
                            subject_X.append(X)
                            subject_y.append(y)

            if len(subject_X) > 0:
                final_X = np.concatenate(subject_X)
                final_y = np.concatenate(subject_y)

                out_x_path = os.path.join(processed_data_dir, f"{subject_folder}_X.npy")
                out_y_path = os.path.join(processed_data_dir, f"{subject_folder}_y.npy")

                np.save(out_x_path, final_X)
                np.save(out_y_path, final_y)
                print(f"Successfully saved {final_X.shape[0]} total windows for {subject_folder}")


# --- נקודת ההרצה של הסקריפט ---
if __name__ == "__main__":

    RAW_DIR = r"C:\Users\ofir2\PycharmProjects\project\project\data\raw"
    PROCESSED_DIR = r"C:\Users\ofir2\PycharmProjects\project\project\data\processed"

    process_all_data(RAW_DIR, PROCESSED_DIR)
