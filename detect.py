import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from collections import defaultdict
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split

# Paths to dataset root
dataset_root = "G:/TU Darmstadt/SEM 1/Practical lab AI/project/data/Olympic Boxing Punch Classification Video Dataset"

# YOLO Model
class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        print("YOLO Model Loaded Successfully!")
    
    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return detections

# LSTM Model for Punch Classification
class PunchLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, output_dim=8):
        super(PunchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Dataset Loader
class BoxingDataset(Dataset):
    def __init__(self, dataset_root, cache_file="processed_data.pkl", train=True, test_size=0.2):
        self.cache_file = os.path.join(dataset_root, cache_file)
        self.label_to_index = {}  

        # Load cached data if available
        if os.path.exists(self.cache_file):
            print("📂 Loading cached dataset...")
            with open(self.cache_file, "rb") as f:
                cached_data = pickle.load(f)

            # Handle old cache formats
            if len(cached_data) == 2:  
                self.data, self.labels = cached_data
                print("⚠️ Old cache format detected. Regenerating label-to-index mapping...")
                self.label_to_index = {label: idx for idx, label in enumerate(set(self.labels))}
                self.labels = [self.label_to_index[label] for label in self.labels]  
                self.save_cache()  
            elif len(cached_data) == 3:  
                self.data, self.labels, self.label_to_index = cached_data
            else:
                raise ValueError("❌ Corrupted cache file! Please delete and reprocess.")
            print(f"✅ Loaded {len(self.data)} samples from cache")
        else:
            print("🚀 Processing videos for the first time...")
            self.data, self.labels = [], []
            self.load_data(dataset_root)
            self.save_cache()

        if len(self.data) == 0:
            print("❌ ERROR: No data loaded. Check dataset structure!")

        self.train = train
        train_data, test_data, train_labels, test_labels = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=42, stratify=self.labels
        )
        self.data = train_data if train else test_data
        self.labels = train_labels if train else test_labels

        print(f"✅ {'Training' if train else 'Testing'} dataset size: {len(self.data)} samples")

    def save_cache(self):
        """Save processed data to a cache file."""
        with open(self.cache_file, "wb") as f:
            pickle.dump((self.data, self.labels, self.label_to_index), f)
        print(f"💾 Cached processed dataset with {len(self.data)} samples!")

    def load_data(self, dataset_root):
        for folder in os.listdir(dataset_root):
            full_path = os.path.join(dataset_root, folder)
            if 'bounding_boxes.json' in os.listdir(os.path.join(full_path, 'data')):
                continue
            annotation_path = os.path.join(dataset_root, folder, "annotations.json")
            bounding_boxes_path = os.path.join(dataset_root, folder, "data", "bounding_boxes.json")
            video_path = os.path.join(dataset_root, folder, "data")
            video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
    
            if not video_files:
                print(f"⚠️ No videos found in {video_path}")
            else:
                print(f"📂 Found {len(video_files)} videos in {folder}: {video_files}")
            
            if not os.path.exists(annotation_path):
                print(f"❌ Missing annotation files in {folder}: Skipping")
                continue

            print(f"📂 Processing {folder} - Found annotations and bounding boxes")
            
            with open(annotation_path, 'r') as f:
                try:
                    annotations = json.load(f)
                except json.JSONDecodeError:
                    print(f"❌ Error parsing JSON in {folder}, skipping...")
                    continue

            if isinstance(annotations, list):
                print("⚠️ Warning: Annotations is a list. Accessing the first element...")
                annotations = annotations[0]  

            if 'tracks' not in annotations:
                print(f"❌ ERROR: 'tracks' key not found in annotations for {folder}")
                continue
            
            # Load video frames
            for video in os.listdir(os.path.join(full_path, 'data')):
                ###video_file = [f for f in os.listdir(video_path) if f.endswith('.mp4')][0]
                cap = cv2.VideoCapture(os.path.join(video_path, video))
                print(f"Processing video: {video}")
                frame_count = 0
                if not cap.isOpened():
                    print("Failed to open video file!")
                else:
                    print("Video file opened successfully.")
                frame_data = {}
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (320,240))
                    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_data[frame_id] = frame
                    frame_count += 1

                num_samples_before = len(self.data)
                # Prepare data
                for track in annotations['tracks']:
                    label = track.get('label', -1)
                    if 'shapes' not in track:
                        print(f"⚠️ No shapes in track {track}, skipping...")
                        continue
                    for shape in track['shapes']:
                        frame_id = shape.get('frame', -1)
                        points = shape.get('points', [])

                        if frame_id in frame_data:
                            self.data.append(points)
                            if label not in self.label_to_index:
                                self.label_to_index[label] = len(self.label_to_index)  

                            self.labels.append(self.label_to_index[label]) 


                del frame

                num_samples_after = len(self.data)
                print(f"✅ Processed {folder} - Loaded {num_samples_after - num_samples_before} new samples")


                if frame_count % 500 == 0:  # Log progress every 500 frames
                    print(f"🔄 Processed {frame_count} frames so far...")
                
                print(f"✅ Loaded {len(self.data)} samples so far")

            cap.release()
            print(f"✅ Finished processing {folder}, Total frames: {frame_count}")

        print(f"✅ Dataset initialized with {len(self.data)} samples")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Ensure labels are integers
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        # Debugging: Check if label is still a string
        if isinstance(self.labels[idx], str):
            print(f"❌ Error: Label '{self.labels[idx]}' at index {idx} is still a string! Converting...")
            self.labels[idx] = self.label_to_index.get(self.labels[idx], -1)

        return data_tensor, label_tensor

# Training Function
def train_lstm(model, train_loader, epochs=3, lr=0.0005):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):  # ✅ Loop over epochs
        total_loss = 0.0  # ✅ Initialize loss
        correct = 0  # ✅ Initialize correct predictions
        total = 0  # ✅ Initialize total samples

        for inputs, labels in train_loader:  # ✅ Iterate over batches
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total  # ✅ Compute accuracy
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "lstm_model.pth")
    print("💾 Model saved as 'lstm_model.pth'!")


def evaluate_lstm(model, test_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(1))
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

            for i in range(min(5, len(labels))):
                print(f"🟢 Actual: {labels[i].item()}, 🔵 Predicted: {preds[i].item()}")

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)  # Added Balanced Accuracy

    # Print results
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ Balanced Accuracy: {balanced_acc:.4f}")  # Printing Balanced Accuracy

    return accuracy, precision, recall, f1, balanced_acc


# Main Execution
if __name__ == "__main__":
    train_dataset = BoxingDataset(dataset_root, train=True)
    test_dataset = BoxingDataset(dataset_root, train=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Train the model
    yolo = YOLODetector()
    lstm_model = PunchLSTM()
    train_lstm(lstm_model, train_loader)

    # Evaluate the model
    lstm_model.load_state_dict(torch.load("lstm_model.pth"))
    lstm_model.eval()
    print("✅ Model loaded successfully!")

    print("📊 Evaluating Model Performance...")
    evaluate_lstm(lstm_model, test_loader) 

  

