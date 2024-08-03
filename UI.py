import os
import pandas as pd
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.utils import to_categorical
from joblib import Parallel, delayed
import streamlit as st

class AudioProcessor:
    def __init__(self, ms=4000, sr=16000, n_mfcc=40):
        self.ms = ms
        self.sr = sr
        self.n_samples = int((sr / 1000) * ms)
        self.n_mfcc = n_mfcc

    def load_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sr)
        # Pad or truncate the signal to the fixed length
        if len(y) > self.n_samples:
            y = y[:self.n_samples]
        else:
            y = np.pad(y, (0, max(0, self.n_samples - len(y))), "constant")
        return y, sr

    def extract_mfcc(self, y):
        mfcc_features = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        return mfcc_features.T.flatten()

    def add_noise(self, y, noise_factor=0.005):
        noise = np.random.randn(len(y))
        augmented_data = y + noise_factor * noise
        return augmented_data

    def process_feature(self, file_path, emotion):
        y, sr = self.load_audio(file_path)
        mfcc = self.extract_mfcc(y)
        return mfcc, emotion

    def process_dataset(self, df, n_jobs=-1):
        paths = df['File_Path'].values
        emotions = df['Emotion'].values

        results = Parallel(n_jobs=n_jobs)(delayed(self.process_feature)(path, emotion) for path, emotion in zip(paths, emotions))

        X, Y = [], []
        for result in results:
            X.append(result[0])
            Y.append(result[1])

        # Pad or truncate sequences to the same length
        max_len = max(len(x) for x in X)
        X = np.array([np.pad(x, (0, max_len - len(x)), 'constant') if len(x) < max_len else x[:max_len] for x in X])

        return X, np.array(Y)

# Prepare the dataset
processor = AudioProcessor()
# df = pd.DataFrame(...)  # Assuming df is your dataframe with 'File_Path' and 'Emotion'
# X, Y = processor.process_dataset(df)

def prepare_dataset(X, Y, val_size=0.1, batch_size=4):
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')

    # Encode the labels
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    joblib.dump(encoder, 'label_encoder.pkl')

    num_classes = len(np.unique(Y))
    Y = to_categorical(Y, num_classes=num_classes)

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    print(type(X_train), type(Y_train), type(X_test), type(Y_test))

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_tensor = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long)
    Y_val_tensor = torch.tensor(np.argmax(Y_val, axis=1), dtype=torch.long)
    Y_test_tensor = torch.tensor(np.argmax(Y_test, axis=1), dtype=torch.long)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor

# X_train, X_val, X_test, Y_train, Y_val, Y_test, X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor = prepare_dataset(X, Y)


# Load the emotion dictionary
emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_emotion_label(file_name):
    if file_name.endswith('.wav'):
        emotion_number = file_name.split('-')[2]
        emotion_label = emotion_dict.get(emotion_number, 'Unknown')
        return pd.DataFrame({'File_Path': [file_name], 'Emotion': [emotion_label]})
    else:
        raise ValueError("File is not a .wav file")

# Load the saved model, scaler, and label encoder
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attention(x)
        return weights

def attentive_statistical_pooling(mfccs, attention_weights):
    weighted_mean = torch.sum(attention_weights * mfccs, dim=1)  # Shape: (batch_size, input_dim)
    weighted_std = torch.sqrt(torch.sum(attention_weights * (mfccs - weighted_mean.unsqueeze(1))**2, dim=1))
    return torch.cat((weighted_mean, weighted_std), dim=1)  # Shape: (batch_size, input_dim * 2)

class TDNN(nn.Module):
    def __init__(self, input_dim, output_dim, context_size, dilation):
        super(TDNN, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, context_size, dilation=dilation)

    def forward(self, x):
        return F.relu(self.conv(x))

class CNN_TDNN_LSTM_ASP_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN_TDNN_LSTM_ASP_Classifier, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # TDNN layers
        self.tdnn1 = TDNN(128, 512, context_size=5, dilation=1)
        self.tdnn2 = TDNN(512, 512, context_size=3, dilation=2)
        self.tdnn3 = TDNN(512, 512, context_size=3, dilation=3)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        
        # Attention layer
        self.attention = Attention(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)  # 256 for mean + 256 for variance
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply CNN layers with batch normalization
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Reshape for TDNN layers
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, height * width, -1).permute(0, 2, 1)  # Change shape to (batch_size, feature_dim, seq_len)
        
        # Apply TDNN layers
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        
        # Prepare input for LSTM
        x = x.permute(0, 2, 1).contiguous()  # Change shape to (batch_size, seq_len, features)
        
        # Apply LSTM layer
        x, _ = self.lstm(x)
        
        # Apply Attention
        attention_weights = self.attention(x)
        x = attentive_statistical_pooling(x, attention_weights)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
# Initialize the model, loss function, and optimizer
input_dim = 5040  # As per your dataset
num_classes = 8
model = CNN_TDNN_LSTM_ASP_Classifier(input_dim=input_dim, num_classes=num_classes)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

scaler = joblib.load('scaler.pkl')
encoder = joblib.load('label_encoder.pkl')

# Streamlit UI
st.title("Audio Emotion Classification")
st.write("Upload a .wav audio file to classify its emotion.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")
if uploaded_file is not None:
    try:
        # Ensure the temporary directory exists
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Save uploaded file to a temporary location
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract emotion label
        df = extract_emotion_label(uploaded_file.name)
        st.write("True Emotion Label:")
        st.write(df)

        # Process the uploaded audio file
        processor = AudioProcessor()
        y, sr = processor.load_audio(temp_file_path)
        mfcc = processor.extract_mfcc(y)
        mfcc = scaler.transform([mfcc])

        # Make prediction
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).view(-1, 1, 70, 72)
        with torch.no_grad():
            outputs = model(mfcc_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = encoder.inverse_transform(predicted.numpy())[0]

        st.write(f"Predicted Emotion Label: {predicted_label}")

        # Clean up the temporary file
        os.remove(temp_file_path)
    except ValueError as e:
        st.error(str(e))
