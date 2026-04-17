import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import config
from utils.dataset import EEGProcessedDataset
from models.spiking_cnn import StandardCNN


def evaluate_and_plot():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # 1. טעינת הנתונים
    test_dir = os.path.join(config.PROCESSED_DIR, "test")
    test_dataset = EEGProcessedDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. הקמת המודל וטעינת המשקולות של "מודל 17"
    model = StandardCNN(num_channels=23).to(device)
    model_path = os.path.join(config.BASE_DIR, "saved_models", "model_17.pth")

    if not os.path.exists(model_path):
        print(f"[!] Error: Could not find {model_path}. Did you move the file?")
        return

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print(f"[*] Evaluating Model 17 on {len(test_dataset)} windows...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. הדפסת התוצאות לטרמינל
    print("\n=== Real Test Results (Model 17) ===")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Seizure']))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # 4. יצירת ושמירת הגרפים
    plots_dir = os.path.join(config.BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)  # יוצר תיקיית plots אם היא לא קיימת

    # ציור מטריצת בלבול
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal (Pred)', 'Seizure (Pred)'],
                yticklabels=['Normal (True)', 'Seizure (True)'])
    plt.title('Confusion Matrix - Model 17')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"\n[*] Visualizations saved to the 'plots' folder!")
    print(f"- Confusion Matrix saved at: {cm_path}")


if __name__ == "__main__":
    evaluate_and_plot()