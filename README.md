WEDA Fall Detection: Video-Based Fall Detection Using Deep Learning 🚀
======================================================================

Welcome to the **WEDA Fall Detection** project! 🎥 This repository implements a deep learning-based system to detect falls from video data using the WEDA Fall Dataset. By leveraging convolutional neural networks (CNNs) like MobileNetV2, this project achieves **100% validation accuracy** in distinguishing Activities of Daily Living (ADLs) from fall events. Perfect for real-time applications in elderly care, hospitals, or smart homes! 🏥💡

Project Overview 🌟
-------------------

This project builds a robust pipeline to detect falls in video footage, addressing the critical need for non-intrusive safety monitoring for the elderly or mobility-impaired. Here's what it does:

1.  **Preprocessing** 📂: Converts videos into frames and resizes them for model compatibility.

2.  **Data Augmentation** 🖼️: Enhances dataset diversity with transformations like flips and rotations.

3.  **Model Training** 🤖: Fine-tunes MobileNetV2 (or ResNet-50) for binary classification (ADLs vs. Falls).

4.  **Evaluation** 📊: Assesses performance with accuracy, confusion matrix, and Grad-CAM visualizations.

5.  **Deployment** 💾: Exports a lightweight model for real-time fall detection.

The result is a highly accurate, interpretable model ready for integration into surveillance systems or IoT devices.

Prerequisites 🛠️
-----------------

To run this project, you'll need:

-   **Hardware** 💻:

    -   GPU (recommended for faster training) or CPU.

    -   At least 16GB RAM and ample storage for video data/frames.

-   **Software** 🐍:

    -   Python 3.8+.

    -   Environment: Google Colab (GPU-enabled) or local setup with PyTorch CUDA support.

-   **Dataset** 📹:

    -   WEDA Fall Dataset with ADLs/ and Falls/ folders containing .MP4 videos.

    -   Place the dataset at /mnt/data/ or update paths in the code.

-   **Dependencies** 📦: Install required libraries:

    ```
    pip install torch torchvision opencv-python numpy albumentations matplotlib seaborn tqdm scikit-learn
    ```

Step-by-Step Execution Guide 🚀
-------------------------------

Follow these steps to set up and run the project:

### 1\. Clone the Repository 📥

Clone this repo to your local machine or cloud environment (e.g., Google Colab):

```
git clone <repository-url>
cd weda-fall-detection
```

### 2\. Prepare the Dataset 📂

-   **Download the WEDA Fall Dataset**: Source it from the official repository or your own copy (see references below).

-   **Organize the Dataset**: Ensure this structure:

    ```
    /mnt/data/
    ├── ADLs/
    │   ├── D01.MP4
    │   ├── D02.MP4
    │   └── ...
    ├── Falls/
    │   ├── F01.MP4
    │   ├── F02.MP4
    │   └── ...
    ```

-   **Update Paths**: Modify input_folder and output_folder in the code if your dataset is stored elsewhere (e.g., /content/data for Colab).

### 3\. Extract Frames from Videos 🎞️

Convert .MP4 videos into frames for processing:

-   **File**: First code cell in ADS CA D17C 47.ipynb.

-   **Purpose**: Extracts every 5th frame, resizes to 224x224 pixels, and saves to /mnt/data/processed_frames/.

-   **Run**:

    ```
    input_folder = "/mnt/data/"
    output_folder = "/mnt/data/processed_frames"
    extract_frames_from_videos(input_folder, output_folder, every_n_frames=5, resize_dim=(224, 224))
    ```

-   **Output**: Frames saved in /mnt/data/processed_frames/ADLs/ and /mnt/data/processed_frames/Falls/.

-   **Expected Output**:

    ```
    [INFO] Extracted 179 frames from F03.MP4 into Falls/
    [INFO] Extracted 129 frames from D05.MP4 into ADLs/
    ...
    ```

### 4\. Augment the Frames 🖌️

Increase dataset diversity with augmentations:

-   **File**: Second code cell in ADS CA D17C 47.ipynb.

-   **Purpose**: Applies horizontal flips, brightness/contrast changes, rotations, and Gaussian noise, generating 3 augmented versions per frame.

-   **Run**:

    ```
    original_frames_dir = "/mnt/data/processed_frames"
    augmented_frames_dir = "/mnt/data/augmented_frames"
    augment_images(original_frames_dir, augmented_frames_dir, num_augmentations=3)
    ```

-   **Output**: Augmented frames saved in /mnt/data/augmented_frames/ADLs/ and /mnt/data/augmented_frames/Falls/.

-   **Expected Output**:

    ```
    [INFO] Data augmentation done. Saved augmented images in: /mnt/data/augmented_frames
    ```

### 5\. Set Up the Model and Data Loaders 🧠

Prepare the dataset and initialize the model:

-   **File**: Third code cell in ADS CA D17C 47.ipynb.

-   **Purpose**: Loads augmented frames with ImageFolder, splits into 80% training/20% validation, and sets up MobileNetV2 for 2-class classification.

-   **Run**:

    ```
    data_dir = "/mnt/data/augmented_frames"
    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    model = get_model("mobilenetv2")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    ```

-   **Note**: Verify GPU availability with torch.cuda.is_available(). CPU fallback is supported but slower.

### 6\. Train the Model 🎓

Train MobileNetV2 and save checkpoints:

-   **File**: Fifth code cell in ADS CA D17C 47.ipynb (updated training loop).

-   **Purpose**: Trains for 4 epochs, saves model checkpoints per epoch, saves the best model, and exports as TorchScript.

-   **Run**:

    ```
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=4)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(trained_model.eval(), dummy_input)
    export_path = "/mnt/data/exported_fall_model.pt"
    traced_model.save(export_path)
    ```

-   **Output**: Saves checkpoints (model_epoch_X.pth), best model (best_model.pth), and exported model (exported_fall_model.pt) in /mnt/data/.

-   **Expected Output**:

    ```
    Epoch 1/4
    ------------------------------
    Training: 100%|██████████| 205/205 [22:28<00:00, 6.58s/it]
    Train Loss: 0.0359, Train Acc: 0.9872
    Validating: 100%|██████████| 52/52 [01:08<00:00, 1.31s/it]
    Val Loss: 0.0003, Val Acc: 1.0000
    [INFO] Model saved: /mnt/data/model_epoch_1.pth
    [INFO] Best model updated and saved!
    ...
    ✅ Training Complete. Best Val Accuracy: 1.0000
    ✅ Exported model saved to: /mnt/data/exported_fall_model.pt
    ```

### 7\. Evaluate the Model 📈

Assess model performance with metrics and visualizations:

-   **File**: Sixth code cell in ADS CA D17C 47.ipynb.

-   **Purpose**: Loads the best model, generates predictions, and computes accuracy, classification report, and confusion matrix.

-   **Run**:

    ```
    model.load_state_dict(torch.load("/mnt/data/best_model.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['ADLs', 'Falls']))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()
    ```

-   **Expected Output**:

    ```
    Accuracy: 1.0000
                  precision    recall  f1-score   support
        ADLs       1.00      1.00      1.00       866
       Falls       1.00      1.00      1.00       773
    accuracy                           1.00      1639
    ```

-   **Visualization**: Displays a confusion matrix plot showing perfect classification.

### 8\. Visualize Results 📷

Generate insightful visualizations:

-   **Accuracy/Loss Curves** 📉: Track training/validation performance.

-   **Sample Predictions** 🖼️: Show frames with true vs. predicted labels.

-   **Grad-CAM Heatmaps** 🔥: Highlight regions the model focuses on for predictions.

-   **Run**: Use matplotlib and seaborn for plotting (see sixth code cell for confusion matrix; extend for other visualizations per the report).

### 9\. Deploy the Model 🚀

Use the exported TorchScript model for real-time inference:

-   Load the model:

    ```
    model = torch.jit.load("/mnt/data/exported_fall_model.pt")
    model.eval()
    ```

-   Process new video frames for fall detection in production environments.

Project Outcomes 🎉
-------------------

-   **Performance**: Achieved **100% validation accuracy** with perfect precision, recall, and F1-scores (1.00) for ADLs and Falls.

-   **Interpretability**: Grad-CAM heatmaps confirm focus on human posture/motion patterns.

-   **Applications**: Ideal for real-time surveillance in elderly care, hospitals, or smart homes.

-   **Scalability**: MobileNetV2's lightweight design suits embedded systems.

Troubleshooting 🔧
------------------

-   **Dataset Path Errors**: Ensure /mnt/data/ matches your setup; update paths as needed.

-   **GPU Issues**: If torch.cuda.is_available() is False, training uses CPU (slower).

-   **Memory Errors**: Reduce batch_size (e.g., to 16) or use a smaller dataset subset.

-   **Library Conflicts**: Use compatible versions (e.g., torch==2.0.0).

References 📚
-------------

-   Marques, J.; Moreno, P. Online Fall Detection Using Wrist Devices. *Sensors* 2023, 23, 1146.

-   Fula, Vanilson & Moreno, Plinio. (2024). Wrist-Based Fall Detection: Towards Generalization across Datasets. *Sensors*. 24. 1679. 10.3390/s24051679.

-   WEDA Fall Dataset documentation (check source for access).

-   PyTorch: https://pytorch.org/

-   OpenCV: https://docs.opencv.org/

Notes 📝
--------

-   Developed on Google Colab with GPU support. Adjust paths for local execution.

-   Extend training beyond 4 epochs by updating num_epochs, but monitor overfitting.

-   To use ResNet-50, change to: model = get_model("resnet50").

This project delivers a **robust, end-to-end solution** for video-based fall detection, ready for real-world safety applications! 🌟
