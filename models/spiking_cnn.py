import torch
import torch.nn as nn

class StandardCNN(nn.Module):
    def __init__(self, num_channels=23):
        super().__init__()

        # שכבת קונבולוציה ראשונה
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)  # מכבה 25% מהמפות כדי למנוע שינון
        )

        # שכבת קונבולוציה שנייה
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)  # מכבה 25% מהמפות
        )

        self.flatten = nn.Flatten()

        # שכבות סיווג (Fully Connected) - הוספנו עומק ו-Dropout אגרסיבי
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 2, 128),  # שכבה חדשה שמכווצת את הנתונים
            nn.ReLU(),
            nn.Dropout(0.5),              # מכבה 50% מהנוירונים כדי להכריח למידה כללית
            nn.Linear(128, 2)             # פלט סופי: 2 מחלקות (נורמלי/התקף)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x