# Audio Emotion Classification

This project aims to classify emotions from audio signals using a combination of Convolutional Neural Networks (CNN), Time-Delay Neural Networks (TDNN), Long Short-Term Memory Networks (LSTM), and Attentive Statistical Pooling (ASP).

## Project Structure

- **AudioProcessor Class**: Handles audio preprocessing, including loading audio files, extracting MFCC features, and adding noise for data augmentation.
- **CNN_TDNN_LSTM_ASP_Classifier Class**: Defines the neural network model combining CNN, TDNN, LSTM, and ASP for feature extraction and classification.

## Steps to Use

1. **Prepare the Dataset**:
    - Use the `AudioProcessor` class to load audio files and extract MFCC features.
    - Split the dataset into training, validation, and test sets.
    - Standardize the features and encode the labels.

2. **Train the Model**:
    - Initialize the `CNN_TDNN_LSTM_ASP_Classifier` model.
    - Train the model using the training and validation sets.
    - Save the best model.

3. **Classify Emotions**:
    - Load the saved model.
    - Use the model to predict emotions from new audio files.

## Usage

1. **Preprocessing**:
    ```python
    processor = AudioProcessor()
    X, Y = processor.process_dataset(df)
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor = prepare_dataset(X, Y)
    ```

2. **Training**:
    ```python
    model = CNN_TDNN_LSTM_ASP_Classifier(input_dim=5040, num_classes=8)
    # Train the model and save the best model
    ```

3. **Prediction**:
    ```python
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    # Predict emotion for new audio files
    ```

## Streamlit UI

The project includes a Streamlit UI for easy interaction:

- Upload a `.wav` file.
- The UI will display the true emotion label and the predicted emotion label.

```python
import streamlit as st

# Streamlit code to upload and classify audio files

## Contributing

Contributions are welcome. Feel free to fork this repository and submit pull requests for improvements or new features. Some data examples are uploaded to try. For more detailed information, refer to the provided PDF presentation.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

