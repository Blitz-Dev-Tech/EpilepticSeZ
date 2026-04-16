import re


def parse_summary(summary_file_path):
    """
    קורא קובץ סיכום של CHB-MIT ומחזיר מילון עם זמני ההתקפים.
    מבנה המילון שחוזר: {'chb01_03.edf': [(2996, 3036)], 'chb01_04.edf': [(1467, 1494)]}
    """
    seizure_map = {}
    current_file = None

    with open(summary_file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        # זיהוי שם הקובץ הנוכחי
        if line.startswith("File Name:"):
            current_file = line.split("File Name:")[1].strip()
            seizure_map[current_file] = []

        # זיהוי שעת התחלה של התקף
        elif line.startswith("Seizure") and "Start Time" in line:
            start_match = re.search(r'Start Time:\s*(\d+)\s*seconds', line)
            if start_match:
                start_time = int(start_match.group(1))

                # אם מצאנו התחלה, הסיום יהיה בוודאות בשורה או שתיים הבאות
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    end_match = re.search(r'End Time:\s*(\d+)\s*seconds', next_line)
                    if end_match:
                        end_time = int(end_match.group(1))
                        seizure_map[current_file].append((start_time, end_time))

    return seizure_map