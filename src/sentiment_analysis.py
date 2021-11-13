from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

def batch(iterable, batchsize=1):
    l = len(iterable)
    for ndx in range(0, l, batchsize):
        yield iterable[ndx:min(ndx + batchsize, l)]

@torch.no_grad()
def predict_sentiment(total_quotes, batchsize=1):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = []
    for quotes in batch(total_quotes, batchsize):
        labels = ["Negative", "Positive"]
        encoded_input = tokenizer(quotes, return_tensors='pt')
        output = model(**encoded_input)
        preds = F.softmax(output.logits)
        for pred in preds.numpy():
            sentiment = labels[np.argmax(pred)]
            sentiments.append(sentiment)
    return sentiments