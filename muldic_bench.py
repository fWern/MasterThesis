import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from dataset_preparation import prepare_muldic
from model import Model, ProjectionHead
from utils import train_muldic_all, evaluate_accuracy_all, evaluate_metrics_all, get_text_features, get_image_features, train_muldic, train_contrastive_muldic, evaluate_accuracy, evaluate_metrics, CustomDataset, CustomDatasetAllFeatures
import random
from transformers import RobertaTokenizer, RobertaModel
import warnings
import argparse

warnings.filterwarnings("ignore", message="Failed to load image Python extension")

device = "cuda" if torch.cuda.is_available() else "cpu"


class CodeFeatureExtractor(nn.Module):
    def __init__(self):
        super(CodeFeatureExtractor, self).__init__()
        self.model = RobertaModel.from_pretrained('microsoft/codebert-base')
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation as the feature
        cls_features = outputs.last_hidden_state[:, 0, :]
        return cls_features

def extract_code_features(codes):
    # Initialize the tokenizer and the model
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = CodeFeatureExtractor().to(device)
    model.eval()

    all_outputs = []
    with torch.no_grad():
        for code in codes:
            encoded_inputs = tokenizer(code, padding=True, truncation=True, return_tensors="pt", max_length=512)
            input_ids = encoded_inputs['input_ids'].to(device)
            attention_mask = encoded_inputs['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(outputs.shape)
            all_outputs.append(outputs.cpu())
            del input_ids, attention_mask, outputs
            torch.cuda.empty_cache()
    concatenated_outputs = torch.cat(all_outputs, dim=0)
    return concatenated_outputs.to(device)

def train_mod(args, input_size, train_loader, test_loader):
    hidden_size = args.hidden_size
    num_classes = 2
    num_epochs = args.num_epochs
    patience = args.patience
    lr = args.learning_rate

    model = Model(input_size, hidden_size, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model = train_muldic_all(model, train_loader, num_epochs, optimizer, criterion, patience, args.robust_training)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    #acc = evaluate_accuracy_all(model, test_loader, args.missing_training, args.missing_rate)
    prec, rec, f1 = evaluate_metrics_all(model, test_loader, args.missing_modality, args.missing_training, args.missing_rate)

    return (prec, rec, f1)

def train_mod_2(args, input_size, train_loader, test_loader):
    hidden_size = args.hidden_size
    num_classes = 2
    num_epochs = args.num_epochs
    patience = args.patience
    lr = args.learning_rate

    model = Model(input_size, hidden_size, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model = train_muldic(model, train_loader, num_epochs, optimizer, criterion, patience)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    #acc = evaluate_accuracy_all(model, test_loader, args.missing_training, args.missing_rate)
    prec, rec, f1 = evaluate_metrics(model, test_loader)

    return (prec, rec, f1)

def train_model(args):
    modality = args.modality
    data_path = args.dataset_dir
    data_label = args.label

    #load data
    df_train_features = pd.read_csv(data_path + args.label + f'/train_data/{data_label}_train_feature.csv')
    df_test_features = pd.read_csv(data_path + args.label + f'/test_data/{data_label}_test_feature.csv')

    df_train_bugs = pd.read_csv(data_path + args.label + f'/train_data/{data_label}_train_bug.csv')
    df_test_bugs = pd.read_csv(data_path + args.label + f'/test_data/{data_label}_test_bug.csv')

    #split train in train and dev
    df_train = pd.concat([df_train_features, df_train_bugs])
    df_test = pd.concat([df_test_features, df_test_bugs])

    train_texts, train_images, train_codes, train_labels = prepare_muldic(df_train, data_path + args.label + '/train_data/images/')
    test_texts, test_images, test_codes, test_labels = prepare_muldic(df_test, data_path + args.label + '/test_data/images/')

    le = LabelEncoder()
    train_labels = torch.tensor(le.fit_transform(train_labels), dtype=torch.long)
    test_labels =  torch.tensor(le.transform(test_labels), dtype=torch.long)

    if not os.path.exists(f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/train_texts.pt'):
        train_texts = get_text_features(train_texts)
        test_texts = get_text_features(test_texts)
        train_images = get_image_features(train_images)
        test_images = get_image_features(test_images)
        train_code_features = extract_code_features(train_codes)
        test_code_features = extract_code_features(test_codes)

        torch.save(train_texts, f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/train_texts.pt')
        torch.save(test_texts, f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/test_texts.pt')
        torch.save(train_texts, f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/train_images.pt')
        torch.save(test_images, f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/test_images.pt')
        torch.save(train_code_features, f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/train_code_features.pt')
        torch.save(test_code_features, f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/test_code_features.pt')
    else:
        train_texts = torch.load(f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/train_texts.pt')
        test_texts = torch.load(f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/test_texts.pt')
        train_images = torch.load(f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/train_images.pt')
        test_images = torch.load(f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/test_images.pt')
        train_code_features = torch.load(f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/train_code_features.pt')
        test_code_features = torch.load(f'/home/felix.wernlein/project/models/muldic_{args.label.lower()}/test_code_features.pt')

    def get_data_loader(train_features, test_features):
        train_dataset = CustomDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = CustomDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        return train_loader, test_loader

    if args.contrastive_learning:
        code_dim = train_code_features.size(1)
        output_dim = 512
        num_epochs_contrastive = args.num_epochs_contrastive
        contrastive_lr = args.learning_rate_contrastive

        projector = ProjectionHead(code_dim, output_dim).to(device)
        total_params = sum(p.numel() for p in projector.parameters())
        print(f"Total number of parameters (Contrastive): {total_params}")
        optimizer_contrastive = optim.AdamW(projector.parameters(), lr=contrastive_lr)

        train_dataset_contrastive = CustomDatasetAllFeatures(train_code_features, train_texts, train_images, train_labels)
        train_loader_contrastive = DataLoader(train_dataset_contrastive, batch_size=args.batch_size_contrastive, shuffle=True)

        projector = train_contrastive_muldic(projector, train_loader_contrastive, num_epochs_contrastive, optimizer_contrastive, args.patience_contrastive, args.top_k)

        # apply learned projection head to code features
        train_code_features = projector(train_code_features)
        test_code_features = projector(test_code_features)
    

    if modality == 'text':
        #train only text
        train_loader, test_loader = get_data_loader(train_texts, test_texts)
        return train_mod_2(args, train_texts.size(1), train_loader, test_loader)
    elif modality == 'image':
        #train only images
        train_loader, test_loader = get_data_loader(train_images, test_images)
        return train_mod_2(args, train_images.size(1), train_loader, test_loader)
    elif modality == 'code':
        #train only codes
        train_loader, test_loader = get_data_loader(train_code_features, test_code_features)
        return train_mod_2(args, train_code_features.size(1), train_loader, test_loader)
    elif modality == 'text_image':
        #train Text image
        train_features = torch.cat((train_texts, train_images), dim=1)
        test_features = torch.cat((test_texts, test_images), dim=1)
        train_loader, test_loader = get_data_loader(train_features, test_features)
        return train_mod_2(args, train_features.size(1), train_loader, test_loader)
    elif modality == 'text_code':
        #train Text Code
        train_features = torch.cat((train_code_features, train_texts), dim=1)
        test_features = torch.cat((test_code_features, test_texts), dim=1)
        train_loader, test_loader = get_data_loader(train_features, test_features)
        return train_mod_2(args, train_features.size(1), train_loader, test_loader)
    elif modality == 'image_code':
        #train Code Image
        train_features = torch.cat((train_code_features, train_images), dim=1)
        test_features = torch.cat((test_code_features, test_images), dim=1)
        train_loader, test_loader = get_data_loader(train_features, test_features)
        return train_mod_2(args, train_features.size(1), train_loader, test_loader)
    elif modality == 'all':
        #train only all three
        train_dataset = CustomDatasetAllFeatures(train_code_features, train_texts, train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = CustomDatasetAllFeatures(test_code_features, test_texts, test_images, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        size = train_code_features.size(1) + train_texts.size(1) + train_images.size(1)
        return train_mod(args, size, train_loader, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuG Bench")
    parser.add_argument('--label', type=str)
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
    #parser.add_argument('--robust_training', action='store_true')
    parser.add_argument('--robust_training', type=str)
    parser.add_argument('--missing_training', action='store_true')
    parser.add_argument('--missing_rate', type=float)
    #parser.add_argument('--missing_modality', type=str)
    parser.add_argument('--missing_modality', nargs='*', default=[], help='List of modalities to be considered missing')

    args = parser.parse_args()
    print(args)

    seed_val = 0
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    prec, rec, f1 = train_model(args)
    print(f"Precision: {prec:.4f} - Recall: {rec:.4f} - F1-Score: {f1:.4f}")