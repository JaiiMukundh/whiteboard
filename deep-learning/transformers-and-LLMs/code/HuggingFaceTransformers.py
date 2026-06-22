from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

''' 
# Easy Pipeline surface level
generator = pipeline("text-generation", "distilgpt2")
generated = generator("It's a sunny day but i still want spicy food")

print(generated)
'''

#Inner Workings of pipeline()

#1. Preprocessing (Tokenization)
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True,       return_tensors="pt")
print(inputs)

#2. The actual model (transformer architecture to encode-decode or do either)
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)

#3. PostProcessing the output (LM Head)
predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
print(predictions)

print(model.config.id2label)

model = AutoModel.from_pretrained("bert-base-cased")
model.save_pretrained("BERT_base_cased")