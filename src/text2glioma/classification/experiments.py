from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from monai.config import print_config
from monai.utils import set_determinism
from monai.networks import nets
from monai.losses import BCEWithLogitsLoss, CEWithLogitsLoss

from ..utils import get_experiment_dataloaders, get_model

def stats(dataloader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def run_experiment(config, experiment_name, exp_type, debug=False, resume=False):
    """Run a classification experiment based on the provided configuration."""
    set_determinism(config.get("seed", 0))
    print_config()
    print(f"Experiment: {experiment_name}, Type: {exp_type}, Debug: {debug}, Resume: {resume}")
    if debug:
        print(f"DEBUG Config: {config}")

    datalist = config.get("datalist", None)
    if datalist is None:
        raise ValueError("Datalist path must be provided in the config.")
    if isinstance(datalist, str):
        with open(datalist, 'r') as f:
            datalist = json.load(f)

    target = datalist.get(exp_type, None) if datalist else None
    if target is None:
        raise ValueError(f"Data for experiment type '{exp_type}' not found in the datalist file {datalist}.")
    
    cache_dir = config.get("cache_dir", "./cache")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = config.get("batch_size", 48)
    num_workers = config.get("num_workers", 4)
    pin_memory = config.get("pin_memory", False)
    no_shuffle = config.get("no_shuffle", False)
    model_type = config.get("model", {}).get("name", "densenet121")

    train_loader, val_loader = get_experiment_dataloaders(
        datalist=datalist,
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=not no_shuffle,
        model_type=model_type,
        ratio=config.get("data_ratio", 0.1),
    )

    model_cfg = config.get("model", None)
    if model_cfg is None:
        raise ValueError("Model configuration is missing.")
    
    model = get_model(model_cfg)
    model = model(**config["params"])
    if resume and Path(config.get("model_save_path", "./model.pth")).exists():
        print(f"Resuming from saved model at {config.get('model_save_path')}")
        model = torch.load(config.get("model_save_path"), map_location="cpu")        
    
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    criterion = config.get("criterion", "CEWithLogitsLoss")
    params = config[criterion].get("params", {}) if criterion in config else {}
    if criterion == "BCEWithLogitsLoss":
        criterion = BCEWithLogitsLoss(**params)
    elif criterion == "CEWithLogitsLoss":
        criterion = CEWithLogitsLoss(**params)
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4), weight_decay=config.get("weight_decay", 1e-5))

    n_epochs = config.get("n_epochs", 100)
    log_dir = Path(config.get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    val_interval = config.get("val_interval", 10)
    best_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data["image"].to(device), data[target].to(device)
            labels = labels.long()
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=config["model"].get("num_classes", 2)).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data["image"].to(device), data[target].to(device)
                    labels = labels.long()
                    if labels.ndim == 1:
                        labels = F.one_hot(labels, num_classes=config["model"].get("num_classes", 2)).float()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                model_save_path = Path(config.get("model_save_path", "./best_model.pth"))
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved best model to {model_save_path} with val loss {best_loss:.4f}")

            writer.add_scalar('Loss/Val', avg_val_loss, epoch)

        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    writer.close()
    model_save_path = Path(config.get("model_save_path", "./final_model.pth"))
    torch.save(model.state_dict, model_save_path)

    print(f"Model saved to {model_save_path}")
    print("Experiment completed.")
    print("Accuracy of final model: {:.3f}%".format(stats(val_loader, model, device)))
