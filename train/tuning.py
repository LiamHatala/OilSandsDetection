import optuna
from torch.utils.data import DataLoader
from torchvision import models, transforms
from transforms.custom_transforms import AddNoise, CropROI_Tensor
from data.oil_sands_dataset import Dataset
from torch.optim import Adam
from metrics.custom_metrics import Calculate_IoU
from models.model import createDeepLabv3
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
 # Add noise as a custom transformation

# Define the function that will train the model and return the validation accuracy or loss
def objective(trial):
    """
    Optuna objective function for training and evaluating a semantic segmentation model
    on the oil sands dataset. Returns a validation F1 score used for hyperparameter optimization.
    """ 

    # Hyperparameter search space
    config = {
    "lr": 1e-02, #8
    "batch_size": 8,
    "rotation_angle": 14.358234476987889,
    "shear_angle": 0.10768166798428605,
    "num_epochs": 5,
    "image_size": (720,1280),
    "noise_std": 0.06967821539400093,
    "blocks": 4,
    "roi": (350, 0, 500, 720)    }

    # setting the values of the hyperparameter
    aug_rotation = config["rotation_angle"] 
    aug_skew = config["shear_angle"] 
    aug_noise_std = config["noise_std"]  # Standard deviation for Gaussian noise
    learning_rate = trial.suggest_categorical("lr",[  1e-5, 1e-4, 1e-3, 1e-2])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    num_epochs = config["num_epochs"]
    blocks = trial.suggest_int("blocks", 1, 5)
    image_size = config["image_size"]
    

    # Modify augmentation based on the trial's hyperparameters
    roi = config["roi"]# x, y, w, h
    transforms_list = [
        transforms.ToTensor(),
        transforms.Resize(size=image_size),
        CropROI_Tensor(roi),

    ]
    
    # Add random rotation
    if aug_rotation > 0:
        transforms_list.append(transforms.RandomRotation(degrees=aug_rotation))
    
    # Add skew (shear)
    if aug_skew > 0:
        transforms_list.append(transforms.RandomAffine(degrees=0, shear=aug_skew))

    transforms_list.append(AddNoise(aug_noise_std))
    
    # Combine all transformations
    transform = transforms.Compose(transforms_list)
    
    
    train_dataset = Dataset(root, img_folder, mask_folder, set_type="train", transforms=transform)
    test_dataset = Dataset(root, img_folder, mask_folder, set_type="test", transforms=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = createDeepLabv3(num_classes=1, layer_blocks_to_train=blocks)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Training loop
    model.train()
    print("Training -  -  -")
    for epoch in range(num_epochs):
        print("Epoch - ", epoch)
        for batch in train_loader:
            inputs = batch["image"].to(device)
            masks = torch.squeeze(batch["mask"].to(device))
            labels = torch.zeros(masks.shape, dtype=torch.long, device=device)
            labels = torch.where(masks > 0, 1, 0)
            labels = labels.permute((0, 1, 2)).to(torch.float32)
            
            optimizer.zero_grad()
        
            outputs = model(inputs)
            outputs = outputs["out"].permute((1, 0, 2, 3))[0, ...]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        print("done epoch ", epoch)
    model.eval()
    val_loss = 0
    alpha = 0.7
    beta = 0.3
    iou = 0
    f1_acc = 0
    print("Validation -  -  -")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            masks = torch.squeeze(batch["mask"].to(device))
            labels = torch.zeros(masks.shape, dtype=torch.long, device=device)
            labels = torch.where(masks > 0, 1, 0)
            labels = labels.permute((0, 1, 2)).to(torch.float32)
            
            outputs = model(inputs)
            outputs = outputs["out"].permute((1, 0, 2, 3))[0, ...]
            y_true = labels
            y_pred = outputs
            
            val_loss += criterion(y_pred, y_true)
            f1_acc += binary_f1_score(y_pred.flatten(),y_true.flatten())
            pred = (torch.sigmoid(y_pred) > 0.5).float()
            iou += Calculate_IoU(pred,y_true)
        
    # Alternative objective value using IOU and loss
    """
    val_loss /= len(test_loader.dataset)
    mean_iou = iou / len(test_loader.dataset)
    objective_value = alpha * val_loss + beta * (1 - mean_iou)
    """
    mean_acc = f1_acc/ len(test_loader.dataset)
    objective_value =  mean_acc
    return objective_value

def save_study_results_to_csv(study, filename="study_results.csv"):
        """
        Saves Optuna study results to a CSV file for analysis and reproducibility.
        """
        # Define the header
        header = ['Trial', 'Objective Value'] + list(study.best_trial.params.keys())
        
        # Open the CSV file for writing
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write header
            
            # Write each trial's parameters and objective value
            for trial in study.trials:
                row = [trial.number, trial.value] + [trial.params.get(param) for param in header[2:]]
                writer.writerow(row)
        print(f"Study results saved to {filename}")
        print(f"Best value: {study.best_value} (params: {study.best_params})")

def optimizeParams():
    """
    Creates and runs an Optuna study for hyperparameter tuning.
    Saves the results and prints the best trial.
    """

    # Run the optimization
    study = optuna.create_study(storage = 'sqlite:///train/optunaStudy/optuna_study_new2.db', study_name = "Oil Sands Segmentation " ,direction = "maximize")
    study.optimize(objective, n_trials=10)

    # Save the results
    save_study_results_to_csv(study)

if __name__ == "__main__":
    root = "/home/khush/code/"
    img_folder = "grouped"
    mask_folder = "labels"
    optimizeParams()
