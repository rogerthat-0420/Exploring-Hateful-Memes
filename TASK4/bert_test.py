import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Example text for classification
text = "This is an example sentence for sentiment analysis."

# Tokenize and convert to tensor
inputs = tokenizer(text, return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Replace 1 with the actual label for your use case

# Forward pass
outputs = model(**inputs, labels=labels)

# Get predictions and probabilities
logits = outputs.logits
probabilities = softmax(logits, dim=1)

# Print the predicted label and probabilities
predicted_label = torch.argmax(probabilities).item()
predicted_class = model.config.id2label[predicted_label]
print(f"Predicted class: {predicted_class}")
print(f"Probabilities: {probabilities.tolist()}")
