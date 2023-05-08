from .model_class import BERTClass, PredDataset
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

model = BERTClass()
#model.load_state_dict(torch.load('bert_1', map_location=torch.device('cpu')))
model.load_state_dict(torch.load(
    'model/bert_1', map_location=torch.device('cpu')))
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', return_dict=False)
device = 'cpu'

pred_params = {'batch_size': 1,
               'shuffle': True,
               'num_workers': 0
               }
MAX_LEN = 200


def predict(sentence):
    output = []
    pred_df = pd.DataFrame({'sentence': [sentence]})
    pred_set = PredDataset(pred_df, tokenizer, MAX_LEN)
    pred_loader = DataLoader(pred_set, **pred_params)
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(pred_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(
                device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            output.append(torch.sigmoid(outputs).cpu().detach().numpy())
    res = np.argmax(output[0][0])
    return sentiment(res)


def sentiment(index):
    if index == 2:
        return 'Positive'
    elif index == 1:
        return 'Neutral'
    elif index == 0:
        return 'Negative'


if __name__ == '__main__':
    print(predict('Apple announces 90% profit'))
