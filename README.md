# Oil Sands Level Detection

This project focuses on detecting the level of oil sands in images using a DeepLabV3+ model. The main goal is to accurately segment the oil sands region from the rest of an image.

### Dependencies

The project relies on several Python libraries. All required packages are listed in the `requirements.txt` file.

**Installation:**
To install the necessary dependencies, navigate to the project's root directory (DeepLabV3Plus-Pytorch) in your terminal and run:
```bash
pip install -r requirements.txt
```
It is highly recommended to use a virtual environment (e.g., venv, conda) to manage project dependencies and avoid conflicts with other Python projects.

The `requirements.txt` file includes packages for:
- Core deep learning tasks (PyTorch, TorchVision, Torcheval)
- Image processing (OpenCV, Pillow)
- Data handling and numerical operations (NumPy, Pandas)
- Hyperparameter optimization (Optuna)
- Visualization (Matplotlib, Plotly, Dash for potential GUI elements)
- And various other utilities.

Ensure you have Python 3.8 or newer installed. Some dependencies, particularly those related to PyTorch and CUDA, may require specific versions of NVIDIA drivers if you plan to use GPU acceleration. Please refer to the official PyTorch installation guide for more details on setting up CUDA support.

### Project Structure

Here's an overview of the main directories and their purpose:

-   `data/`: Contains scripts for dataset handling (`oil_sands_dataset.py`) and is the recommended location for storing your image and mask datasets (see "Dataset Preparation" for the expected structure).
-   `models/`: Includes the definition of the DeepLabV3+ model architecture (`model.py`).
-   `network/`: Contains lower-level network components for DeepLabV3+, such as backbone definitions (e.g., ResNet, MobileNetV2) and the core DeepLab implementation.
-   `train/`: Houses scripts related to model training and optimization.
    -   `train.py`: Main script for training the model.
    -   `tuning.py`: Script for hyperparameter tuning using Optuna.
    -   `bestModels/`: Intended for storing your best performing trained model weights.
    -   `exp_runs/`: Default directory for saving training logs (e.g., `log.csv`).
    -   `optunaStudy/`: Stores Optuna's study database (e.g., `optuna_study_new2.db`).
-   `transforms/`: Contains custom data augmentation and transformation pipelines (`custom_transforms.py`).
-   `utils/`: Holds utility scripts.
    -   `visualizer.py`: Script for visualizing model predictions on videos.
-   `metrics/`: Defines custom evaluation metrics (`custom_metrics.py`).
-   `chkpts/`: Contains pre-trained model checkpoints that can be used as a starting point or for specific model configurations.
-   `gui/`: Includes files for a graphical user interface (e.g., `start_gui.py`), likely for interactive testing or demonstration.
-   `requirements.txt`: Lists all Python dependencies for the project.
-   `README.md`: This file, providing an overview and guide to the project.

### Dataset Preparation

**Data Source:**
- The dataset used for training and evaluation can be obtained from [Link to dataset or description of how to access it].
- Alternatively, if you have your own dataset, ensure it is in a compatible format. Images should be in standard formats (e.g., JPG, PNG), and corresponding masks should be grayscale images where pixel values represent class labels.

**Directory Structure:**
Organize your dataset in the following structure within the `data` directory:
```
data/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   ├── val/
│   │   ├── image3.jpg
│   │   └── ...
│   └── test/
│       ├── image4.jpg
│       └── ...
└── masks/
    ├── train/
    │   ├── image1_mask.png
    │   ├── image2_mask.png
    │   └── ...
    ├── val/
    │   ├── image3_mask.png
    │   └── ...
    └── test/
        ├── image4_mask.png
        └── ...
```
- `images/train/`: Contains training images.
- `images/val/`: Contains validation images.
- `images/test/`: Contains test images.
- `masks/train/`: Contains corresponding masks for training images.
- `masks/val/`: Contains corresponding masks for validation images.
- `masks/test/`: Contains corresponding masks for test images.

Make sure the filenames of images and their corresponding masks match (e.g., `image1.jpg` and `image1_mask.png`).

### 1. Training the model
Make sure you are in the main directory - DeepLabV3Plus-Pytorch and then run the following command:
```python
python3 -m train.train
```

**Configuration:**
Training parameters are configured directly within the `train/train.py` script, inside the `if __name__ == "__main__":` block. Key parameters in the `config` dictionary include:
- `lr`: Learning rate
- `batch_size`: Batch size for training and testing
- `num_epochs`: Number of training epochs
- `image_size`: Target size for input images (height, width)
- `noise_std`: Standard deviation for added noise (data augmentation)
- `rotation_angle`: Maximum rotation angle for data augmentation
- `shear_angle`: Maximum shear angle for data augmentation
- `roi`: Region of Interest to crop from the images `(top, left, height, width)`
- `blocks`: Number of ResNet blocks to unfreeze and train in the DeepLabV3 backbone.

Modify these values in the script to suit your dataset and training requirements.

**Training Logs:**
- Progress, loss, and metrics (F1 score, AUROC, level accuracy) are printed to the console during training for each epoch and phase (train/test).
- A detailed log is saved in CSV format to `exp_runs/log.csv` (relative to the main project directory). This file contains:
    - `epoch`: Current epoch number.
    - `train_loss`: Training loss for the epoch.
    - `test_loss`: Validation/test loss for the epoch.
    - `train_<metric>`: Value of each metric (e.g., `train_f1_score`) on the training set.
    - `test_<metric>`: Value of each metric (e.g., `test_f1_score`) on the validation/test set.

**Trained Models:**
- After training completes, the script saves the model weights to a file named `best_mdl_wts.pt` in the main project directory (`DeepLabV3Plus-Pytorch/`).
- It appears the convention is to manually move preferred models into the `train/bestModels/` directory for safekeeping or later use, as suggested by the `get_Pretrained_model` function in `train/train.py`.

### 2. Hyperparameter tuning
Make sure you are in the main directory - DeepLabV3Plus-Pytorch and then run the following command to start the Optuna hyperparameter search:
```python
python3 -m train.tuning
```

**Search Space:**
The `train/tuning.py` script uses Optuna to find the best hyperparameters. The current search space defined in the `objective` function includes:
- `learning_rate`: Searched over `[1e-5, 1e-4, 1e-3, 1e-2]`.
- `batch_size`: Searched over `[8, 16, 32, 64]`.
- `blocks`: Integer values from 1 to 5, representing the number of ResNet blocks to unfreeze and train in the DeepLabV3 backbone.

Other parameters like data augmentation details (`rotation_angle`, `shear_angle`, `noise_std`), number of epochs for each trial (`num_epochs`), `image_size`, and `roi` are currently fixed within the `objective` function in `train/tuning.py`. To change them or add them to the search space, you would need to modify the script.

**Resuming Tuning:**
- The Optuna study is configured to use SQLite for storage: `sqlite:///train/optunaStudy/optuna_study_new2.db` with the study name `"Oil Sands Segmentation "`.
- If the tuning process is interrupted or you want to run more trials, simply re-running the `python3 -m train.tuning` command will resume the study from where it left off, using the existing database.

**Analyzing Tuning Results:**
- After the Optuna study completes (or is interrupted), the script calls `save_study_results_to_csv(study)`. This function saves a detailed summary of all trials to a file named `study_results.csv` in the main project directory. This CSV file includes:
    - `Trial`: The trial number.
    - `Objective Value`: The metric being optimized (currently F1 score on the validation set).
    - Columns for each hyperparameter in the search space (e.g., `lr`, `batch_size`, `blocks`).
- The script also prints the best objective value and the corresponding best hyperparameters found during the study to the console.
- For more advanced analysis and visualization, the Optuna database `train/optunaStudy/optuna_study_new2.db` can be used with tools like `optuna-dashboard` or by loading the study in a Python script using `optuna.load_study()`.

### 3. Visualization of the results
Make sure you are in the main directory - DeepLabV3Plus-Pytorch. Before running, you will need to edit `utils/visualizer.py` to set the `video_pth` variable to the path of your input video. You might also want to change `model_pth` if you are not using the default `best_mdl_wts.pt`.

Then, run the following command:
```python
python3 -m utils.visualizer
```

**Output of the Script:**
The visualization script performs several actions:
1.  **Real-time Display:** It opens a window titled "Oil Sands Level Classification" showing processed video frames.
    *   A blue rectangle indicates the Region of Interest (ROI) where level detection is performed.
    *   Two horizontal red lines are drawn on the frame, representing a band around the detected oil sands level.
    *   The detected level is displayed as a percentage (e.g., "Level: 75%") on the frame. This percentage indicates the fill level from the bottom of the ROI.
    *   The calculated percentage is also printed to the console for each processed frame.
2.  **Annotated Video:** It saves an output video file named `output.mp4` in the main project directory. This video contains the same frames and annotations as shown in the real-time display.
3.  **Frame Skipping:** By default, the script processes approximately one frame every 10 seconds of video (skipping 300 frames at a time for a 30fps video). This is to speed up visualization of long videos.

**Customization:**
Most customizations require editing the `utils/visualizer.py` script:
-   **Input Video (`video_pth`):** Essential. Change this variable in the script to your video file.
-   **Model Path (`model_pth`):** Defaults to `best_mdl_wts.pt`. Change this if you want to use a different trained model.
-   **Region of Interest (`x, y, w, h`):** The coordinates for the ROI are hardcoded within the `display_video` function (currently `(350, 0, 500, 720)`). Modify these values to focus the level detection on a different part of the video.
-   **Frame Processing Rate:** Change the `frame_index += 300` line in `display_video` to process more or fewer frames. For example, `frame_index += 1` would process every frame.
-   **Output Video Name:** The output file is hardcoded as `output.mp4`. Change this in the `cv2.VideoWriter` line if needed.
-   **Level Detection Logic:** Parameters within the `get_level` function (e.g., threshold for mask binarization, morphological operation parameters, contour analysis logic) can be adjusted for different conditions.
-   **Visualization Details:** Colors, line thickness, font type, and text position for annotations are hardcoded using OpenCV functions (`cv2.line`, `cv2.putText`, etc.) and can be modified.

**Keyboard Controls (during real-time display):**
-   `q`: Quit the visualization.
-   `p`: Pause or resume the video processing.
-   `r`: Rewind the video by a short interval (currently set to skip back by 30 frames from current position).

### Contributing

Contributions to this project are welcome! If you have suggestions for improvements, bug fixes, or new features, please consider the following:

1.  **Reporting Bugs:** If you encounter a bug, please open an issue on the project's issue tracker (if available). Describe the bug in detail, including steps to reproduce it, the expected behavior, and the actual behavior.
2.  **Suggesting Enhancements:** For feature requests or enhancements, please also open an issue to discuss your ideas.
3.  **Pull Requests:** If you'd like to contribute code:
    *   Fork the repository.
    *   Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name` or `bugfix/your-bug-fix`).
    *   Make your changes and commit them with clear, descriptive messages.
    *   Ensure your code adheres to any existing coding style or guidelines.
    *   Push your branch to your fork (`git push origin feature/your-feature-name`).
    *   Open a pull request against the main repository, explaining the changes you've made.

Please note that this is a general guideline. Specific contribution processes may be defined by the project maintainers.

### License

The licensing information for this project has not yet been specified. Please check for a `LICENSE` file in the repository or contact the project maintainers for clarification. Until a license is explicitly stated, it is advisable to assume the software is proprietary and not available for use, modification, or distribution without express permission from the copyright holders.
