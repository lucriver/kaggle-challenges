# Use GPU
import os
import logging
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split, StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--log-identifier', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--weight-decay', type=float)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--k-folds', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--weights-identifier', type=str)
parser.add_argument('--data-dir', type=str)

args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler(f'./data/logs/{args.log_identifier}.log',mode='w'),
        logging.StreamHandler()
    ]
)

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

df_train = pd.read_csv(f'{args.data_dir}/train.csv')
df_test = pd.read_csv(f'{args.data_dir}/test.csv')

def get_encodeds(texts,tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

def predict(texts,model,tokenizer,device):
    encoded_input = get_encodeds(texts,tokenizer)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions

def get_dataloader(x: list, y: list, tokenizer, batch_size: int):
    encodeds = get_encodeds(x,tokenizer)
    x_input_ids = encodeds['input_ids']
    x_attn_masks = encodeds['attention_mask']
    y = torch.tensor(y)
    dataset = TensorDataset(x_input_ids, x_attn_masks, y)
    return DataLoader(dataset, batch_size=batch_size)

X = df_train['text'].copy()
y = df_train['label'].copy()

# training parameters
hyperparams = {
    'epochs': args.epochs,
    'lr': args.lr,
    'weight_decay': args.weight_decay,
    'batch_size': args.batch_size
}


if args.k_folds:
    logging.info(f"Hyperparameters: {hyperparams}.")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    train_fold_losses = defaultdict(list)
    val_fold_losses = defaultdict(list)
    logging.info('Starting k-folds cross validation...')
    for i, (train_index, test_index) in enumerate(skf.split(X,y)):
        logging.info(f'Fold: {i+1},')

        torch.cuda.empty_cache()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        train_dataloader = get_dataloader(X[train_index].values.tolist(),y[train_index].values, tokenizer, hyperparams['batch_size'])
        test_dataloader = get_dataloader(X[test_index].values.tolist(),y[test_index].values, tokenizer,  hyperparams['batch_size'])

        logging.info('Defining model...')
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.hidden_dropout_prob = 0.3  # Default is usually 0.1
        config.attention_probs_dropout_prob = 0.3  # Default is usually 0.1
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=hyperparams['lr'],
                                      weight_decay=hyperparams['weight_decay'])

        for epoch in range(hyperparams['epochs']):
            model.train()
            total_train_loss = 0
            for batch in train_dataloader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_train_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_fold_losses[i].append(total_train_loss / len(train_dataloader))
            logging.info(f"Training loss after epoch {epoch + 1}: {total_train_loss / len(train_dataloader)}")

            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for batch in test_dataloader:
                    input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()
                val_fold_losses[i].append(total_val_loss / len(test_dataloader))
                logging.info(f"Validation loss after epoch {epoch + 1}: {total_val_loss / len(test_dataloader)}")

                accuracies = []
                for batch in test_dataloader:
                    input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2]
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=1)
                    score = accuracy_score(labels.numpy(), predictions.cpu().numpy())
                    accuracies.append(score)
                logging.info(f"Epoch {epoch + 1} accuracy: {np.mean(accuracies)}")


    for fold in range(5):
        logging.info(f"min loss for training fold {fold}: {min(train_fold_losses[fold])}")
    logging.info(f"Average minimum training loss: {np.mean([min(losses) for losses in train_fold_losses.values()])}")

    for fold in range(5):
        logging.info(f"min loss for validation fold {fold}: {min(val_fold_losses[fold])}")
    logging.info(f"Average minimum validation loss: {np.mean([min(losses) for losses in val_fold_losses.values()])}")


if args.train:
    logging.info(f"Hyperparameters: {hyperparams}.")
    logging.info('Starting training...')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataloader = get_dataloader(X.values.tolist(),y.values, tokenizer, hyperparams['batch_size'])

    logging.info('Defining model...')
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.hidden_dropout_prob = 0.3  # Default is usually 0.1
    config.attention_probs_dropout_prob = 0.3  # Default is usually 0.1
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hyperparams['lr'],
                                  weight_decay=hyperparams['weight_decay'])
    
    train_losses = []
    for epoch in range(hyperparams['epochs']):
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(total_train_loss / len(train_dataloader))
        logging.info(f"Training loss after epoch {epoch + 1}: {total_train_loss / len(train_dataloader)}")

    if args.weights_identifier:
        torch.save(model.state_dict(), f'./data/model_weights_{args.weights_identifier}')