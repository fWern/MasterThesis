import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from utils import train_all, evaluate_accuracy_all, evaluate_metrics_all, get_text_features, get_image_features_video, evaluate_accuracy, train, train_contrastive, evaluate_metrics, CustomDataset, CustomDatasetAllFeatures, device
from audio_utils import extract_features_wav
from model import Model, ProjectionHead
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import argparse


def train_mod(args, input_size, train_loader, test_loader, dev_loader, num_classes):
        hidden_size = args.hidden_size
        num_epochs = args.num_epochs
        patience = args.patience
        lr = args.learning_rate
        
        model = Model(input_size, hidden_size, num_classes).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model = train_all(model, train_loader, dev_loader, num_epochs, optimizer, criterion, patience, args.robust_training)
        acc = evaluate_accuracy_all(model, test_loader, args.missing_modality, args.missing_training, args.missing_rate)
        prec, rec, f1 = evaluate_metrics_all(model, test_loader, args.missing_modality, args.missing_training, args.missing_rate)

        del model
        del optimizer
        del criterion
        with torch.no_grad():
            torch.cuda.empty_cache()
            
        return (acc, f1)


def train_mod_2(args, input_size, train_loader, test_loader, dev_loader, num_classes):
        hidden_size = args.hidden_size
        num_epochs = args.num_epochs
        patience = args.patience
        lr = args.learning_rate
        
        model = Model(input_size, hidden_size, num_classes).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model = train(model, train_loader, dev_loader, num_epochs, optimizer, criterion, patience)
        acc = evaluate_accuracy(model, test_loader)
        prec, rec, f1 = evaluate_metrics(model, test_loader)

        del model
        del optimizer
        del criterion
        with torch.no_grad():
            torch.cuda.empty_cache()
            
        return (acc, f1)

def train_model(args):
    modality = args.modality
    data = pd.read_csv(args.dataset_dir + "label.csv")
    
    # frames text to array
    data['frames'] = data['frames'].apply(lambda x: x.split(',') if pd.notna(x) else [])
    data['face_frames'] = data['face_frames'].apply(lambda x: x.split(',') if pd.notna(x) else [])

    if not os.path.exists(f'/home/felix.wernlein/project/models/{args.dataset}/train_texts.pt'):
        train_texts = get_text_features(data[data['mode'] == 'train']['text'].values)
        test_texts = get_text_features(data[data['mode'] == 'test']['text'].values)
        dev_texts = get_text_features(data[data['mode'] == 'valid']['text'].values)
        
        train_images = get_image_features_video(data[data['mode'] == 'train']['frames'])
        test_images = get_image_features_video(data[data['mode'] == 'test']['frames'])
        dev_images = get_image_features_video(data[data['mode'] == 'valid']['frames'])

        torch.save(train_texts, f'/home/felix.wernlein/project/models/{args.dataset}/train_texts.pt')
        torch.save(test_texts, f'/home/felix.wernlein/project/models/{args.dataset}/test_texts.pt')
        torch.save(dev_texts, f'/home/felix.wernlein/project/models/{args.dataset}/dev_texts.pt')
        torch.save(train_images, f'/home/felix.wernlein/project/models/{args.dataset}/train_images.pt')
        torch.save(test_images, f'/home/felix.wernlein/project/models/{args.dataset}/test_images.pt')
        torch.save(dev_images, f'/home/felix.wernlein/project/models/{args.dataset}/dev_images.pt')
    else:
        train_texts = torch.load(f'/home/felix.wernlein/project/models/{args.dataset}/train_texts.pt')
        test_texts = torch.load(f'/home/felix.wernlein/project/models/{args.dataset}/test_texts.pt')
        dev_texts = torch.load(f'/home/felix.wernlein/project/models/{args.dataset}/dev_texts.pt')

        train_images = torch.load(f'/home/felix.wernlein/project/models/{args.dataset}/train_images.pt')
        test_images = torch.load(f'/home/felix.wernlein/project/models/{args.dataset}/test_images.pt')
        dev_images = torch.load(f'/home/felix.wernlein/project/models/{args.dataset}/dev_images.pt')

        train_images_face = torch.load(f'/home/felix.wernlein/project/models/{args.dataset}/train_images_face.pt')
        test_images_face = torch.load(f'/home/felix.wernlein/project/models/{args.dataset}/test_images_face.pt')
        dev_images_face = torch.load(f'/home/felix.wernlein/project/models/{args.dataset}/dev_images_face.pt')

    if modality == 'audio' or modality == 'text_audio' or modality == 'image_audio' or modality == 'all':
        train_audios = extract_features_wav(data[data['mode'] == 'train']['audio']).to(device)
        test_audios = extract_features_wav(data[data['mode'] == 'test']['audio']).to(device)
        dev_audios = extract_features_wav(data[data['mode'] == 'valid']['audio']).to(device)
    
    # get labels
    if args.label == 'acc7':
        le = LabelEncoder()
        train_labels = torch.tensor(le.fit_transform(data[data['mode'] == 'train']['label']), dtype=torch.long)
        test_labels = torch.tensor(le.transform(data[data['mode'] == 'test']['label']), dtype=torch.long)
        dev_labels = torch.tensor(le.transform(data[data['mode'] == 'valid']['label']), dtype=torch.long)
    elif args.label == 'acc2':
        le = LabelEncoder()
        train_labels = torch.tensor(le.fit_transform(data[data['mode'] == 'train']['annotation']), dtype=torch.long)
        test_labels = torch.tensor(le.transform(data[data['mode'] == 'test']['annotation']), dtype=torch.long)
        dev_labels = torch.tensor(le.transform(data[data['mode'] == 'valid']['annotation']), dtype=torch.long)
    
    def get_data_loader(train_features, test_features, dev_features):
        train_dataset = CustomDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = CustomDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        dev_dataset = CustomDataset(dev_features, dev_labels)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

        return train_loader, test_loader, dev_loader
    
    num_classes = len(le.classes_)
    

    if args.contrastive_learning:
        audio_dim = train_audios.size(1)
        output_dim = 512
        num_epochs_contrastive = args.num_epochs_contrastive
        batch_size = args.batch_size_contrastive
        lr_contrastive = args.learning_rate_contrastive

        projector = ProjectionHead(audio_dim, output_dim).to(device)
        total_params = sum(p.numel() for p in projector.parameters())
        print(f"Total number of parameters: {total_params}")
        
        optimizer_contrastive = optim.AdamW(projector.parameters(), lr=lr_contrastive)

        train_dataset_contrastive = CustomDatasetAllFeatures(train_audios, train_texts, train_images, train_labels)
        train_loader_contrastive = DataLoader(train_dataset_contrastive, batch_size=batch_size, shuffle=True)

        dev_dataset_contrastive = CustomDatasetAllFeatures(dev_audios, dev_texts, train_images, dev_labels)
        dev_loader_contrastive = DataLoader(dev_dataset_contrastive, batch_size=batch_size)

        projector = train_contrastive(projector, train_loader_contrastive, dev_loader_contrastive, num_epochs_contrastive, optimizer_contrastive, args.patience_contrastive, args.top_k)
        
        # apply learned projection head to code features
        train_audios = projector(train_audios)
        test_audios = projector(test_audios)
        dev_audios = projector(dev_audios)


    if modality == 'text':
        #train texts
        train_loader, test_loader, dev_loader = get_data_loader(train_texts, test_texts, dev_texts)
        return train_mod_2(args, train_texts.size(1), train_loader, test_loader, dev_loader, num_classes)
    elif modality == 'image':
        train_loader, test_loader, dev_loader = get_data_loader(train_images, test_images, dev_images)
        return train_mod_2(args, train_images.size(1), train_loader, test_loader, dev_loader, num_classes)
    elif modality == 'audio':
        train_loader, test_loader, dev_loader = get_data_loader(train_audios, test_audios, dev_audios)
        return train_mod_2(args, train_audios.size(1), train_loader, test_loader, dev_loader, num_classes)
    elif modality == 'text_image':
        train_features = torch.cat((train_texts, train_images), dim=1)
        test_features = torch.cat((test_texts, test_images), dim=1)
        dev_features = torch.cat((dev_texts, dev_images), dim=1)
        train_loader, test_loader, dev_loader = get_data_loader(train_features, test_features, dev_features)
        return train_mod_2(args, train_features.size(1), train_loader, test_loader, dev_loader, num_classes)
    elif modality == 'text_audio':
        train_features = torch.cat((train_texts, train_audios), dim=1)
        test_features = torch.cat((test_texts, test_audios), dim=1)
        dev_features = torch.cat((dev_texts, dev_audios), dim=1)
        train_loader, test_loader, dev_loader = get_data_loader(train_features, test_features, dev_features)
        return train_mod_2(args, train_features.size(1), train_loader, test_loader, dev_loader, num_classes)
    elif modality == 'image_audio':
        print(1)
        train_features = torch.cat((train_images, train_audios), dim=1)
        test_features = torch.cat((test_images, test_audios), dim=1)
        dev_features = torch.cat((dev_images, dev_audios), dim=1)
        train_loader, test_loader, dev_loader = get_data_loader(train_features, test_features, dev_features)
        return train_mod_2(args, train_features.size(1), train_loader, test_loader, dev_loader, num_classes)
    elif modality == 'all':
        train_dataset = CustomDatasetAllFeatures(train_audios, train_texts, train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = CustomDatasetAllFeatures(test_audios, test_texts, test_images, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        dev_dataset = CustomDatasetAllFeatures(dev_audios, dev_texts, dev_images, dev_labels)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
        size = train_audios.size(1) + train_texts.size(1) + train_images.size(1)
        return train_mod(args, size, train_loader, test_loader, dev_loader, num_classes)


def compute_statistics(results):
    results_array = np.array(results)
    
    acc = results_array[:, 0]
    f1 = results_array[:, 1]
    
    mean_acc = np.mean(acc)
    std_acc = np.std(acc)

    mean_f1 = np.mean(f1)
    std_f1 = np.std(f1)
    
    return mean_acc, std_acc, mean_f1, std_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mosi/Mosei Bench")
    parser.add_argument('--label', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--modality', type=str)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--iterations', type=int)

    #contrastive learning
    parser.add_argument('--contrastive_learning', action='store_true', help='Enable contrastive learning')
    parser.add_argument('--learning_rate_contrastive', type=float)
    parser.add_argument('--batch_size_contrastive', type=int)
    parser.add_argument('--num_epochs_contrastive', type=int)
    parser.add_argument('--patience_contrastive', type=int)
    parser.add_argument('--top_k', type=int)

    #robustness training
    parser.add_argument('--robust_training', type=str)
    parser.add_argument('--missing_training', action='store_true')
    parser.add_argument('--missing_rate', type=float)
    parser.add_argument('--missing_modality', nargs='*', default=[], help='List of modalities to be considered missing')

    #pcag
    #parser.add_argument('--pcag_dim', type=int)
    #parser.add_argument('--pcag_batch_size', type=int)
    #parser.add_argument('--pcag_learning_rate', type=float)

    args = parser.parse_args()
    print(args)

    seed_val = 0
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    acc, f1 = train_model(args)
    print(f"Accuracy: {acc:.4f} - F1-Score: {f1:.4f}")
