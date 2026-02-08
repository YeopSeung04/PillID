import pandas as pd
import matplotlib.pyplot as plt

csv_path = "../runs/detect/train4/results.csv"
df = pd.read_csv(csv_path)

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train/cls_loss"], label="train_cls")
plt.plot(df["epoch"], df["val/cls_loss"], label="val_cls")
plt.plot(df["epoch"], df["train/box_loss"], label="train_box")
plt.plot(df["epoch"], df["val/box_loss"], label="val_box")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("YOLO Char Detection Training Loss")
plt.grid()
plt.tight_layout()
plt.savefig("../runs/detect/train4/custom_loss.png")
plt.show()
