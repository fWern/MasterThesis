import PIL.Image
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr
import random

torch.cuda.empty_cache()
PIL.Image.MAX_IMAGE_PIXELS = None

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        return features, labels


class CustomDatasetAllFeatures(Dataset):
    def __init__(self, features_1, features_2, features_3, labels):
        self.features_1 = features_1
        self.features_2 = features_2
        self.features_3 = features_3
        self.labels = labels

    def __len__(self):
        return len(self.features_1)

    def __getitem__(self, idx):
        features_1 = self.features_1[idx]
        features_2 = self.features_2[idx]
        features_3 = self.features_3[idx]
        labels = self.labels[idx]
        return features_1, features_2, features_3, labels
    

class CustomDatasetTwoFeatures(Dataset):
    def __init__(self, features_1, features_2, labels):
        self.features_1 = features_1
        self.features_2 = features_2
        self.labels = labels

    def __len__(self):
        return len(self.features_1)

    def __getitem__(self, idx):
        features_1 = self.features_1[idx]
        features_2 = self.features_2[idx]
        labels = self.labels[idx]
        return features_1, features_2, labels


def get_text_features(texts):
    text_features = []
    for idx in tqdm(range(len(texts))):
        try:
            text = texts[idx]
            text = clip.tokenize(text, truncate=True).to(device)
            with torch.no_grad():
                text_feature = clip_model.encode_text(text).to(dtype=torch.float32)
            text_features.append(text_feature)
        except:
            text_features.append(torch.zeros(512).unsqueeze(0).to(device))
    return torch.cat(text_features)

def get_image_features(images):
    image_features = []
    for idx in tqdm(range(len(images))):
        try:
            image = images[idx]
            #image = image.convert('RGBA')
            image = clip_preprocess(Image.open(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = clip_model.encode_image(image).to(dtype=torch.float32)
            image_features.append(image_feature)
        except:
            image_features.append(torch.zeros(512).unsqueeze(0).to(device))
    return torch.cat(image_features)

def get_image_features_video(all_frames):
    image_features = []
    for frames in tqdm(all_frames):
        if not frames:
            image_features.append(torch.zeros(512).to(device))
            continue
            
        images = [clip_preprocess(Image.open(image)) for image in frames]
        images = torch.stack(images).to(device)

        with torch.no_grad():
            frame_features = clip_model.encode_image(images).to(dtype=torch.float32)
        pooled_features, _ = torch.max(frame_features, dim=0)
        image_features.append(pooled_features)

    return torch.stack(image_features)


class EarlyStopping:
    def __init__(self, patience=20, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_score = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train(model, train_loader, dev_loader, num_epochs, optimizer, criterion, patience):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=1e-6)
    best_val = float('inf')
    early_stopping = EarlyStopping(patience=patience)
    best_acc = 0
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features_train, labels in train_loader:
            features_train = features_train.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(features_train)
            loss = criterion(output, labels)

            loss.backward(retain_graph=True)
            optimizer.step()
            torch.cuda.empty_cache()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features_dev, labels_dev in dev_loader:
                features_dev = features_dev.to(device)
                labels_dev = labels_dev.to(device)
                output = model(features_dev)
                loss = criterion(output, labels_dev)
                val_loss += loss.item()
                val_acc = evaluate_accuracy(model, dev_loader)

        train_loss /= len(train_loader)
        val_loss /= len(dev_loader)
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc= val_acc
            best_model = model
        #if (epoch+1) % 20 == 0:
        #    print("val acc: ", val_acc)
        #    print(f"Epoch [{epoch + 1}/{num_epochs}]")
        #    print(f"  Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            #print(f"Early Stop epoch: {epoch}")
            return best_model
    return best_model

def perturb_data_with_noise(modality, missing_percentage, noise_percentage):
    num_samples, num_features = modality.shape

    num_zeroed = int(missing_percentage * num_samples)
    num_noisy = int(noise_percentage * num_samples)

    zeroed_indices = random.sample(range(num_samples), num_zeroed)
    remaining_indices = list(set(range(num_samples)) - set(zeroed_indices))
    noisy_indices = random.sample(remaining_indices, num_noisy)

    mod_clone = modality.clone()

    mod_clone[zeroed_indices, :] = 0

    noise = torch.randn_like(mod_clone[noisy_indices, :])
    mod_clone[noisy_indices, :] += noise

    return mod_clone.to(device)

def perturb_data(modality, missing_percentage):
    num_samples, _ = modality.shape
    num_noisy = int(missing_percentage * num_samples)
    noisy_indices = random.sample(range(num_samples), num_noisy)
    mod_clone = modality.clone()
    # Zero out entire rows
    mod_clone[noisy_indices, :] = 0
    return mod_clone

def train_all(model, train_loader, dev_loader, num_epochs, optimizer, criterion, patience, robust_training):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=1e-6)
    best_val = float('inf')
    early_stopping = EarlyStopping(patience=patience)
    best_acc = 0
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for tab, text, image, labels in train_loader:
            tab = tab.to(device)
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            if robust_training == 'text':
                text = perturb_data_with_noise(text, 0.15, 0.15)
            elif robust_training == 'all':
                text = perturb_data_with_noise(text, 0.15, 0.15)
                image = perturb_data_with_noise(image, 0.15, 0.15)
                tab = perturb_data_with_noise(tab, 0.15, 0.15)
            
            optimizer.zero_grad()

            features_train = torch.cat((tab, text, image), dim=1)

            output = model(features_train)
            loss = criterion(output, labels)

            loss.backward(retain_graph=True)
            optimizer.step()
            torch.cuda.empty_cache()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for dev_tab, dev_text, dev_image, labels_dev in dev_loader:
                dev_tab = dev_tab.to(device)
                dev_text = dev_text.to(device)
                dev_image = dev_image.to(device)
                labels_dev = labels_dev.to(device)

                features_dev = torch.cat((dev_tab, dev_text, dev_image), dim=1)
                labels_dev = labels_dev.to(device)
                output = model(features_dev)
                loss = criterion(output, labels_dev)
                val_loss += loss.item()
                val_acc = evaluate_accuracy_all(model, dev_loader)

        train_loss /= len(train_loader)
        val_loss /= len(dev_loader)
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc= val_acc
            best_model = model
        #if (epoch+1) % 20 == 0:
        #    print("val acc: ", val_acc)
        #    print(f"Epoch [{epoch + 1}/{num_epochs}]")
        #    print(f"  Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            #print(f"Early Stop epoch: {epoch}")
            return best_model
    return best_model

def train_muldic(model, train_loader, num_epochs, optimizer, criterion, patience):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    best_train_loss = float('inf')
    early_stopping = EarlyStopping(patience=patience)
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features_train, labels in train_loader:
            features_train = features_train.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(features_train)
            loss = criterion(output, labels)

            loss.backward(retain_graph=True)
            optimizer.step()
            torch.cuda.empty_cache()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step(train_loss)

        # Optionally, implement early stopping based on training loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model = model

        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            return best_model

    return best_model

def train_muldic_all(model, train_loader, num_epochs, optimizer, criterion, patience, robust_training):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    best_train_loss = float('inf')
    early_stopping = EarlyStopping(patience=patience)
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for code, text, image, labels in train_loader:
            code = code.to(device)
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            if robust_training == 'text':
                text = perturb_data_with_noise(text, 0.15, 0.15)
            elif robust_training == 'all':
                text = perturb_data_with_noise(text, 0.15, 0.15)
                image = perturb_data_with_noise(image, 0.15, 0.15)
                code = perturb_data_with_noise(code, 0.15, 0.15)

            optimizer.zero_grad()
            features_train = torch.cat((code, text, image), dim=1)
            output = model(features_train)
            loss = criterion(output, labels)

            loss.backward(retain_graph=True)
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step(train_loss)

        # Optionally, implement early stopping based on training loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model = model

        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            return best_model

    return best_model

def evaluate_accuracy(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_feature, test_labels in test_loader:
            test_feature = test_feature.to(device)
            test_labels = test_labels.to(device)
            output = model(test_feature)
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(probs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    #print(f"Accuracy: {accuracy:.3f}")
    return accuracy

def evaluate_accuracy_all(model, test_loader, missing_modality=None, missing_training=False, missing_rate=0.0):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for tab, text, image, test_labels in test_loader:
            tab = tab.to(device)
            text = text.to(device)
            image = image.to(device)
            test_labels = test_labels.to(device)
            if missing_training:
                if 'text' in missing_modality:
                    text = perturb_data(text, missing_rate)
                if 'image' in missing_modality: 
                    image = perturb_data(image, missing_rate)
                if 'tab' in missing_modality:
                    tab = perturb_data(tab, missing_rate)
                if 'audio' in missing_modality: 
                    tab = perturb_data(tab, missing_rate)
                    
            test_features = torch.cat((tab, text, image), dim=1)
            test_labels = test_labels.to(device)
            output = model(test_features)
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(probs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    #print(f"Accuracy: {accuracy:.3f}")
    return accuracy



def evaluate_log_loss(model, test_loader, all_labels):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for test_feature, test_labels in test_loader:
            test_feature = test_feature.to(device)
            test_labels = test_labels.to(device)
            output = model(test_feature)
            all_outputs.append(output.cpu())
            all_targets.append(test_labels.cpu())

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    probs = F.softmax(all_outputs, dim=1).numpy()

    log = log_loss(all_targets.numpy(), probs, labels=all_labels)
    #print(f"Log Loss: {log:.3f}")
    return log

def evaluate_log_loss_all(model, test_loader, all_labels, missing_modality=None, missing_training=False, missing_rate=0.0):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for tab, text, image, test_labels in test_loader:
            tab = tab.to(device)
            text = text.to(device)
            image = image.to(device)
            test_labels = test_labels.to(device)

            if missing_training:
                if 'text' in missing_modality:
                    text = perturb_data(text, missing_rate)
                if 'image' in missing_modality: 
                    image = perturb_data(image, missing_rate)
                if 'tab' in missing_modality:
                    tab = perturb_data(tab, missing_rate)

            test_features = torch.cat((tab, text, image), dim=1)
            output = model(test_features)
            all_outputs.append(output.cpu())
            all_targets.append(test_labels.cpu())

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    probs = F.softmax(all_outputs, dim=1).numpy()

    log = log_loss(all_targets.numpy(), probs, labels=all_labels)
    #print(f"Log Loss: {log:.3f}")
    return log


def evaluate_metrics_all(model, loader, missing_modality=None, missing_training=False, missing_rate=0.0):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for tab, text, image, test_labels in loader:
            tab = tab.to(device)
            text = text.to(device)
            image = image.to(device)
            test_labels = test_labels.to(device)

            if missing_training:
                if 'text' in missing_modality:
                    text = perturb_data(text, missing_rate)
                if 'image' in missing_modality: 
                    image = perturb_data(image, missing_rate)
                if 'tab' in missing_modality:
                    tab = perturb_data(tab, missing_rate)
                if 'audio' in missing_modality:
                    tab = perturb_data(tab, missing_rate)
                if 'code' in missing_modality:
                    tab = perturb_data(tab, missing_rate)

            test_features = torch.cat((tab, text, image), dim=1)

            outputs = model(test_features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    #print(f"Precision: {precision:.4f}")
    #print(f"Recall: {recall:.4f}")
    #print(f"F1 Score: {f1:.4f}")
    return precision, recall, f1

def evaluate_metrics(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    #print(f"Precision: {precision:.4f}")
    #print(f"Recall: {recall:.4f}")
    #print(f"F1 Score: {f1:.4f}")
    return precision, recall, f1

class NTXent(nn.Module):
    def __init__(self, t=0.07, top_k=5):
        super(NTXent, self).__init__()
        self.t = t
        self.top_k = top_k

    def forward(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        cos_sim = torch.exp((x1 @ x2.T) / self.t)
        batch_size = cos_sim.size(0)

        pos_sim = torch.diag(cos_sim)

        # Mask to exclude positive pairs from the negative set
        neg_mask = torch.ones_like(cos_sim, device=device) - torch.eye(batch_size, device=device)

        # Calculate negative similarities and apply mask
        neg_sim = cos_sim * neg_mask

        k = min(self.top_k, batch_size - 1)
        if k > 0:
            top_k_neg_sim, _ = torch.topk(neg_sim, k, dim=1)
        else:
            top_k_neg_sim = neg_sim

        # Sum of top_k hardest negatives for each sample
        neg_sum = top_k_neg_sim.sum(dim=1)

        # Calculate the NTXent loss
        loss = -torch.log(pos_sim / (neg_sum + 1e-7))
        return loss.mean()


def train_contrastive(model, train_loader, dev_loader, num_epochs, optimizer, patience, top_k):
    ntxent = NTXent(top_k=top_k)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.8)
    early_stop = EarlyStopping(patience)
    best_val = float('inf')
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for tab_features, text_features, image_features, labels in train_loader:
            tab_features = tab_features.to(device)
            text_features = text_features.to(device)
            image_features = image_features.to(device)

            tab_features = model(tab_features)

            optimizer.zero_grad()
            loss_text = ntxent(tab_features, text_features)
            loss_image = ntxent(tab_features, image_features)
            loss = (loss_text + loss_image) / 2.0
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for dev_tab_features, dev_text_features, dev_image_features, dev_label in dev_loader:
                dev_tab_features = dev_tab_features.to(device)
                dev_text_features = dev_text_features.to(device)
                dev_image_features = dev_image_features.to(device)

                dev_tab_features = model(dev_tab_features)

                loss_dev_tab_text = ntxent(dev_tab_features, dev_text_features)
                loss_dev_tab_image = ntxent(dev_tab_features, dev_image_features)

                v_loss = (loss_dev_tab_text + loss_dev_tab_image) / 2.0
                val_loss += v_loss.item()

        total_loss /= len(train_loader)
        val_loss /= len(dev_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_model = model

        #if (epoch+1) % 50 == 0:
        #    print(f"Epoch: {epoch+1} - Loss: {total_loss:.3f} - Validation Loss: {val_loss:.3f}")

        early_stop(val_loss, model)
        if early_stop.early_stop:
            #print(f"Early Stop epoch: {epoch+1}")
            #print(f"Epoch: {epoch + 1} - Loss: {total_loss:.3f} - Validation Loss: {val_loss:.3f}")
            return best_model
    return best_model


def train_contrastive_muldic(model, train_loader, num_epochs, optimizer, patience, top_k):
    ntxent = NTXent(top_k=top_k)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    early_stop = EarlyStopping(patience)
    best_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for tab_features, text_features, image_features, labels in train_loader:
            tab_features = tab_features.to(device)
            text_features = text_features.to(device)
            image_features = image_features.to(device)

            tab_features = model(tab_features)

            optimizer.zero_grad()
            loss_text = ntxent(tab_features, text_features)
            loss_image = ntxent(tab_features, image_features)
            loss = (loss_text + loss_image) / 2.0
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader)
        scheduler.step(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            best_model = model

        early_stop(total_loss, model)
        if early_stop.early_stop:
            return best_model