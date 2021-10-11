from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch


# this function input the list of ele,
# and ele include a sample of  three tensors
# [token_tensor, segmentation_tensor, label_tensor]
def create_format_input(input):
    tokens_tensors = [ele[0] for ele in input]
    segmentation_tensors = [ele[1] for ele in input]

    if input[0][2] is not None:
        label_ids = torch.stack([ele[2] for ele in input])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first= True, padding_value = 0)
    # pad_sequence: the default length is max_len in this batch
    segmentation_tensors = pad_sequence(segmentation_tensors, batch_first= True, padding_value = 0)

    masked_tensors = torch.zeros(tokens_tensors.shape, dtype = torch.long)
    masked_tensors = masked_tensors.masked_fill(tokens_tensors != 0,1)

    return tokens_tensors, segmentation_tensors, masked_tensors, label_ids


def create_batch_data(traindata):
    batch_size = 64
    trainloader = DataLoader(traindata, batch_size= batch_size, collate_fn=create_format_input)
    yield trainloader


def create_test_batch_data():
    # construct the dataset for prediction
    test_dataset = CreateDataset('test', tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=256, collate_fn=create_format_input)
    predictions = get_predictions(model, test_dataloader)
    index_map = {v : k for k, v in test_dataset.label_map.items()}

    df = pd.Dataframe({'Category': predictions.to_list()})
    df['Category'] = df['Category'].apply(lambda x: index_map(x))


