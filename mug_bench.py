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
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
from autogluon.core.constants import MULTICLASS
from dataset_preparation import prepare_df
from model import Model, ProjectionHead
from utils import CustomDatasetAllFeatures, evaluate_accuracy_all, evaluate_log_loss_all, train_all, get_text_features, get_image_features, train, evaluate_accuracy, evaluate_log_loss, train_contrastive, CustomDataset, CustomDatasetAllFeatures, device
import warnings
import random
import argparse

warnings.filterwarnings("ignore")


def train_mod(args, input_size, train_loader, test_loader, dev_loader, num_classes, all_labels):
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
    log = evaluate_log_loss_all(model, test_loader, all_labels, args.missing_modality, args.missing_training, args.missing_rate)

    return (acc, log)


def get_tab_features(df_train, df_test, df_dev, train_labels, col_label):
    auto_ml = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False,
        enable_text_ngram_features=False,
        enable_vision_features=False,
    )
    auto_ml.fit(df_train.drop(columns=col_label))
    train_features = auto_ml.transform(df_train.drop(columns=col_label))
    test_features = auto_ml.transform(df_test.drop(columns=col_label))
    dev_features = auto_ml.transform(df_dev.drop(columns=col_label))

    tab_model = TabularNeuralNetTorchModel()
    tab_model.problem_type = MULTICLASS
    tab_model.quantile_levels = None
    tab_model._set_default_params()
    params = tab_model._get_model_params()
    processor_kwargs, optimizer_kwargs, fit_kwargs, loss_kwargs, params = tab_model._prepare_params(params)

    tab_model._preprocess_set_features(train_features)

    # to create processor
    train_dataset, _ = tab_model._generate_datasets(train_features, train_labels, processor_kwargs)

    train_tensor = torch.tensor(tab_model.processor.transform(train_features), dtype=torch.float32).to(device)
    test_tensor = torch.tensor(tab_model.processor.transform(test_features), dtype=torch.float32).to(device)
    dev_tensor = torch.tensor(tab_model.processor.transform(dev_features), dtype=torch.float32).to(device)
    return train_tensor, test_tensor, dev_tensor

def train_model(args):
    modality = args.modality
    df_train = pd.read_csv(args.dataset_dir + "/train.csv")
    df_test = pd.read_csv(args.dataset_dir + "/test.csv")
    df_dev = pd.read_csv(args.dataset_dir + "/dev.csv")

    df_train, df_test, df_dev = prepare_df(df_train, df_test, df_dev, args.label, args.dataset_dir)
    
    le = LabelEncoder()
    train_labels = le.fit_transform(df_train[args.label])
    test_labels = le.transform(df_test[args.label])
    dev_labels = le.transform(df_dev[args.label])

    # can save the extracted features for text and images, and also tabular, 
    # since output of CLIP and tabular features extractor are deterministic
    if not os.path.exists(f'/home/felix.wernlein/project/models/mug_{args.dataset}/train_texts.pt'):
        texts_train = list(df_train['combined_text'])
        images_train = list(df_train['IMAGE PATH'])
        texts_test = list(df_test['combined_text'])
        images_test = list(df_test['IMAGE PATH'])
        texts_dev = list(df_dev['combined_text'])
        images_dev = list(df_dev['IMAGE PATH'])
        
        train_tab, test_tab, dev_tab = get_tab_features(df_train, df_test, df_dev, train_labels, args.label)

        train_texts = get_text_features(texts_train)
        train_images = get_image_features(images_train)
        test_texts = get_text_features(texts_test)
        test_images = get_image_features(images_test)
        dev_texts = get_text_features(texts_dev)
        dev_images = get_image_features(images_dev)

        torch.save(train_texts, f'/home/felix.wernlein/project/models/mug_{args.dataset}/train_texts.pt')
        torch.save(test_texts, f'/home/felix.wernlein/project/models/mug_{args.dataset}/test_texts.pt')
        torch.save(dev_texts, f'/home/felix.wernlein/project/models/mug_{args.dataset}/dev_texts.pt')
        torch.save(train_images, f'/home/felix.wernlein/project/models/mug_{args.dataset}/train_images.pt')
        torch.save(test_images, f'/home/felix.wernlein/project/models/mug_{args.dataset}/test_images.pt')
        torch.save(dev_images, f'/home/felix.wernlein/project/models/mug_{args.dataset}/dev_images.pt')
        torch.save(train_tab, f'/home/felix.wernlein/project/models/mug_{args.dataset}/train_tab.pt')
        torch.save(test_tab, f'/home/felix.wernlein/project/models/mug_{args.dataset}/test_tab.pt')
        torch.save(dev_tab, f'/home/felix.wernlein/project/models/mug_{args.dataset}/dev_tab.pt')
    else:
        train_texts = torch.load(f'/home/felix.wernlein/project/models/mug_{args.dataset}/train_texts.pt')
        test_texts = torch.load(f'/home/felix.wernlein/project/models/mug_{args.dataset}/test_texts.pt')
        dev_texts = torch.load(f'/home/felix.wernlein/project/models/mug_{args.dataset}/dev_texts.pt')

        train_images = torch.load(f'/home/felix.wernlein/project/models/mug_{args.dataset}/train_images.pt')
        test_images = torch.load(f'/home/felix.wernlein/project/models/mug_{args.dataset}/test_images.pt')
        dev_images = torch.load(f'/home/felix.wernlein/project/models/mug_{args.dataset}/dev_images.pt')

        train_tab = torch.load(f'/home/felix.wernlein/project/models/mug_{args.dataset}/train_tab.pt')
        test_tab = torch.load(f'/home/felix.wernlein/project/models/mug_{args.dataset}/test_tab.pt')
        dev_tab = torch.load(f'/home/felix.wernlein/project/models/mug_{args.dataset}/dev_tab.pt')

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    dev_labels = torch.tensor(dev_labels, dtype=torch.long)

    #train model
    def get_data_loader(train_features, test_features, dev_features):
        train_dataset = CustomDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = CustomDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        dev_dataset = CustomDataset(dev_features, dev_labels)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

        return train_loader, test_loader, dev_loader
    
    all_labels = np.unique(np.concatenate([train_labels.numpy(), test_labels.numpy(), dev_labels.numpy()]))
    num_classes = len(le.classes_)

    if args.contrastive_learning:        
        dim = train_tab.size(1)
        output_dim = 512
        num_epochs_contrastive = 500
        batch_size = args.batch_size_contrastive
        contrastive_lr = args.learning_rate_contrastive

        projector = ProjectionHead(dim, output_dim).to(device)
        total_params = sum(p.numel() for p in projector.parameters())
        print(f"Total number of parameters (Contrastive): {total_params}")
        optimizer_contrastive = optim.AdamW(projector.parameters(), lr=contrastive_lr)

        train_dataset_contrastive = CustomDatasetAllFeatures(train_tab, train_texts, train_images, train_labels)
        train_loader_contrastive = DataLoader(train_dataset_contrastive, batch_size=batch_size, shuffle=True)

        dev_dataset_contrastive = CustomDatasetAllFeatures(dev_tab, dev_texts, dev_images, dev_labels)
        dev_loader_contrastive = DataLoader(dev_dataset_contrastive, batch_size=batch_size)

        projector = train_contrastive(projector, train_loader_contrastive, dev_loader_contrastive, num_epochs_contrastive, optimizer_contrastive, args.patience_contrastive, args.top_k)
        # apply learned projection head to code features
        train_tab = projector(train_tab)
        test_tab = projector(test_tab)
        dev_tab = projector(dev_tab)
    

    if modality == 'text':
        #train only text
        train_loader, test_loader, dev_loader = get_data_loader(train_texts, test_texts, dev_texts)
        return train_mod(args, train_texts.size(1), train_loader, test_loader, dev_loader, num_classes, all_labels)
    elif modality == 'image':
        #train only images
        train_loader, test_loader, dev_loader = get_data_loader(train_images, test_images, dev_images)
        return train_mod(args, train_images.size(1), train_loader, test_loader, dev_loader, num_classes, all_labels)
    elif modality == 'tab':
        #train only tab
        train_loader, test_loader, dev_loader = get_data_loader(train_tab, test_tab, dev_tab)
        return train_mod(args, train_tab.size(1), train_loader, test_loader, dev_loader, num_classes, all_labels)
    elif modality == 'text_image':
        #train Text image
        train_features = torch.cat((train_texts, train_images), dim=1)
        test_features = torch.cat((test_texts, test_images), dim=1)
        dev_features = torch.cat((dev_texts, dev_images), dim=1)
        train_loader, test_loader, dev_loader = get_data_loader(train_features, test_features, dev_features)
        return train_mod(args, train_features.size(1), train_loader, test_loader, dev_loader, num_classes, all_labels)
    elif modality == 'text_tab':
    #train Text tab
        train_features = torch.cat((train_tab, train_texts), dim=1)
        test_features = torch.cat((test_tab, test_texts), dim=1)
        dev_features = torch.cat((dev_tab, dev_texts), dim=1)
        train_loader, test_loader, dev_loader = get_data_loader(train_features, test_features, dev_features)
        return train_mod(args, train_features.size(1), train_loader, test_loader, dev_loader, num_classes, all_labels)
    elif modality == 'image_tab':
        #train tab Image
        train_features = torch.cat((train_tab, train_images), dim=1)
        test_features = torch.cat((test_tab, test_images), dim=1)
        dev_features = torch.cat((dev_tab, dev_images), dim=1)
        train_loader, test_loader, dev_loader = get_data_loader(train_features, test_features, dev_features)
        return train_mod(args, train_features.size(1), train_loader, test_loader, dev_loader, num_classes, all_labels)
    elif modality == 'all':
        #train only all three
        train_dataset = CustomDatasetAllFeatures(train_tab, train_texts, train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = CustomDatasetAllFeatures(test_tab, test_texts, test_images, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        dev_dataset = CustomDatasetAllFeatures(dev_tab, dev_texts, dev_images, dev_labels)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
        size = train_tab.size(1) + train_texts.size(1) + train_images.size(1)
        return train_mod(args, size, train_loader, test_loader, dev_loader, num_classes, all_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuG Bench")
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
    #parser.add_argument('--robust_training', action='store_true')
    parser.add_argument('--robust_training', type=str)
    parser.add_argument('--missing_training', action='store_true')
    parser.add_argument('--missing_rate', type=float)
    #parser.add_argument('--missing_modality', type=str)
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
    acc, log = train_model(args)
    print(f"Accuracy: {acc:.4f} - Log Loss: {log:.4f}")

    
