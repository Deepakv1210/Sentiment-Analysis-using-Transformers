import torch
import transformers
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn as nn
class_name = ['negative','neutral','positive']

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class TransformerSentimentClassifier(nn.Module):
    def __init__(self, n_classes, hidden_size=768, num_attention_heads=12, num_layers=12):
        super(TransformerSentimentClassifier, self).__init__()
            
        self.bert_config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_layers,
            output_attentions=True 
        )
        
        self.bert = BertModel(config=self.bert_config)

        for param in self.bert.parameters():
            param.requires_grad = True
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['last_hidden_state'][:, 0, :] 
        logits = self.fc(pooled_output)
        return logits

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TransformerSentimentClassifier(n_classes=3, hidden_size=512, num_attention_heads=4, num_layers=1)

model.load_state_dict(torch.load('best_model_final_0.8663023575188831.bin'))

def predict_sentiment(model, tokenizer, review_text, max_len=512):
    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=max_len,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    return class_name[prediction]
