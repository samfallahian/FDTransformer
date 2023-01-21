from transformers import BertModel
import torch

# Instantiate the BERT model
bert = BertModel.from_pretrained('bert-base-uncased')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.layer1 = nn.Linear(768, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self
