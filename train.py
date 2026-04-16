# train.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# ייבוא הקבצים שלנו
import config
from utils.dataset import CHBMITDataset, build_global_summary_dict

# !!! חשוב: כאן אתה מייבא את המודל שלך !!!
# שנה את "StandardCNN" לשם של המחלקה שמוגדרת בקובץ הרשת שלך
from models.spiking_cnn import StandardCNN


def load_datasets(json_filename, global_summary):
    """
    פונקציה שטוענת רשימת קבצים מ-JSON, יוצרת Dataset לכל קובץ,
    ומחברת את כולם למחסן נתונים אחד ענק.
    """
    json_path = os.path.join(config.SPLITS_DIR, json_filename)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[!] Please run utils/data_splitter.py first! Could not find {json_path}")

    with open(json_path, 'r') as f:
        file_list = json.load(f)

    datasets = []
    for rel_path in file_list:
        full_path = os.path.join(config.DATA_DIR, rel_path)
        file_name = os.path.basename(rel_path)

        # משיכת זמני ההתקף מהמילון הגלובלי (או רשימה ריקה אם אין)
        seizure_times = global_summary.get(file_name, [])

        # יצירת ה-Dataset (טוען רק כותרות בשלב זה)
        ds = CHBMITDataset(
            edf_path=full_path,
            seizure_times=seizure_times,
            window_size_sec=config.WINDOW_SIZE_SEC
        )

        # מוסיף רק אם הקובץ נטען בהצלחה ויש בו חלונות
        if len(ds) > 0:
            datasets.append(ds)

    # מחבר את כולם יחד באופן וירטואלי
    return ConcatDataset(datasets)


def main():
    # 1. הגדרת כרטיס מסך (GPU) או מעבד (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 2. בניית המילון שמכיל את כל ההתקפים
    print("[*] Scanning summary files to build seizure map...")
    summary_dict = build_global_summary_dict(config.DATA_DIR)

    # 3. טעינת הנתונים לאימון ומבחן
    print("[*] Loading datasets... (This maps the files, doesn't load them into RAM yet)")
    train_dataset = load_datasets('train_files.json', summary_dict)
    test_dataset = load_datasets('test_files.json', summary_dict)

    print(f"[*] Total Train Windows (4-sec each): {len(train_dataset)}")
    print(f"[*] Total Test Windows (4-sec each): {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 4. הגדרת המודל
    # ודא שה-CNN שלך מצפה ל-23 ערוצים בכניסה!
    model = StandardCNN(num_channels=23).to(device)

    # התמודדות עם חוסר איזון: על כל 100 חלונות נורמליים יש אולי חלון התקף אחד.
    # אנחנו אומרים לרשת: "אם את מפספסת התקף, העונש שלך חמור פי 10!"
    class_weights = torch.tensor([1.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 5. לופ האימון
    print("[*] Starting Training Loop...")
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # העברת הנתונים לכרטיס המסך/מעבד
            inputs, labels = inputs.to(device), labels.to(device)

            # איפוס גרדיאנטים
            optimizer.zero_grad()

            # הרצת הנתונים קדימה ברשת
            outputs = model(inputs)

            # חישוב השגיאה
            loss = criterion(outputs, labels)

            # עדכון המשקולות
            loss.backward()
            optimizer.step()

            # סטטיסטיקות
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # הדפסה כל 50 באצ'ים
            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{config.EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # סיכום של האפוק
        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"=== Epoch {epoch + 1} Completed | Avg Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% ===")


if __name__ == "__main__":
    main()