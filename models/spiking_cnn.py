# models/spiking_cnn.py
import torch
import torch.nn as nn


class StandardCNN(nn.Module):
    def __init__(self, num_channels=23):
        super().__init__()

        # שכבת קונבולוציה ראשונה - עכשיו Conv2d כי אנחנו עובדים על "תמונת" STFT
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # מקטין את התמונה פי 2
        )

        # שכבת קונבולוציה שנייה
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # מקטין שוב פי 2
        )

        self.flatten = nn.Flatten()

        # --- איך חישבנו את 4096? ---
        # 1. הקלט המקורי לכל ערוץ: 129 (תדרים) על 9 (זמן)
        # 2. אחרי MaxPool ראשון: 129/2 = 64, 9/2 = 4 (גודל 64x4)
        # 3. אחרי MaxPool שני: 64/2 = 32, 4/2 = 2 (גודל 32x2)
        # 4. כופלים במספר הערוצים שיצאו מהשכבה השנייה (64)
        # 64 * 32 * 2 = 4096
        self.fc = nn.Linear(64 * 32 * 2, 2)

    def forward(self, x):
        # x נכנס במבנה של: [Batch, Channels, Frequencies, Time]
        # למשל: [32, 23, 129, 9]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x  # פלט בגודל 2 (סיכוי לנורמלי לעומת סיכוי להתקף)
