from ultralytics import YOLO
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Load model
model = YOLO('yolov8n.pt')

def main():
    # Train the model
    model.train(data='Dataset/SplitData/dataoffline.yaml', epochs=10)

    # Evaluate on validation set
    results = model.val()
    
    # Get predictions and ground truth
    preds = []
    targets = []

    # Use the validation dataloader
    val_loader = model.dataloader['val']
    
    for batch in val_loader:
        imgs = batch['img'].to(model.device)
        labels = batch['cls'].cpu(). numpy()  # Ground truth

        with torch.no_grad():
            outputs = model(imgs)
            for output in outputs:
                pred_classes = output.boxes.cls.cpu().numpy()
                preds.extend(pred_classes)

        targets.extend(labels)

    # Convert to NumPy arrays
    preds = np.array(preds, dtype=int)
    targets = np.array(targets, dtype=int)

    # Confusion Matrix
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # Heatmap version (if you want a more detailed look)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

    # Classification report (precision, recall, F1-score, etc.)
    print("Classification Report:\n")
    print(classification_report(targets, preds))

if __name__ == '__main__':
    main()
