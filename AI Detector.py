import numpy as np
import pandas as pd
import torch
import re
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from time import time
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def tokenize(text):
    tokenized_text = re.findall(r'\b\w+\b', text.lower())
    return tokenized_text

def encode_text(text, vocab):
    tokenized_text = tokenize(text)
    encoded_text = [vocab.get(word, vocab['<unk>']) for word in tokenized_text]
    return encoded_text

def pad_or_truncate(encoded_text, max_len):
    if len(encoded_text) > max_len:
        return encoded_text[:max_len]
    else:
        return encoded_text + [vocab['<pad>']] * (max_len - len(encoded_text))

class SimpleNNWithEmbeddingXL(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(SimpleNNWithEmbeddingXL, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 800)
        self.fc1 = nn.Linear(800, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        return x

class SimpleNNWithEmbeddingL(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(SimpleNNWithEmbeddingL, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 500)
        self.fc1 = nn.Linear(500, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

class SimpleNNWithEmbeddingN(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(SimpleNNWithEmbeddingN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.fc1 = nn.Linear(300, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        start = time()
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        end = time()
        length = end-start
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average CE Loss: {avg_loss:.4f}, Time: {length:.2f}s")

def get_predictions_and_probabilities(model,test_loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            predicted_labels = torch.argmax(outputs, dim=1)
            all_labels.extend(predicted_labels.cpu().numpy())
    return all_probs, all_labels

def confusion_report(model, test_loader, model_name):
    pred_probs, pred_labels = get_predictions_and_probabilities(model, test_loader)
    fp, tn, tp, fn = confusion_matrix(y_tensor_test, pred_labels).ravel()
    print(model_name)
    print("Confusion_matrix:\n", [round(float(fn/(fn+tp)), 5), round(float(tp/(tp+fn)), 5)], "\n", [round(float(tn/(tn+fp)), 5), round(float(fp/(fp+tn)), 5)])
    #print(confusion_matrix(y_tensor_test, pred_labels))
    print("-----------")
    print(classification_report(y_tensor_test, pred_labels))
    print("-----------")

def classify_text(model, text):
    encoded_text = pad_or_truncate(encode_text(text, vocab), max_len=Max_Seq_Length)
    input_tensor = torch.tensor(encoded_text).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred_prob = F.softmax(output, dim=1)
        predicted_label = torch.argmax(output, dim=1).item()
    if predicted_label == 0:
        print(f'The model predicts that this piece of text is {pred_prob[0][0].item()*100}% Human-generated!')
    elif predicted_label == 1:
        print(f'The model predicts that this piece of text is {pred_prob[0][1].item()*100:.4f}% AI-generated!')


while True:
    model_type = input("Enter model type (N/L/XL): ")
    if model_type.upper() != "N" and model_type.upper() != "L" and model_type.upper() != "XL":
        print("Invalid model type. Please enter N, L, or XL.")
        continue
    else:
        model_type = model_type.upper()
        break

while True:
    train = input("Enter training mode (True/False): ")
    if train.lower() not in ['true', 'false']:
        print("Invalid input. Please enter True or False.")
        continue
    else:
        train = train.lower() == 'true'
        if train:
            while True:
                epochs = int(input("Enter number of epochs: "))
                if epochs > 0:
                    save_path = f"models/AI_detector{epochs}{model_type}.pth"
                    break
                else:
                    print("Invalid input. Please enter a positive integer.")
        else:
            while True:
                epochs = int(input("Enter number of epochs to load: "))
                if epochs > 0:
                    save_path = f"models/AI_detector{epochs}{model_type}.pth"
                    break
                else:
                    print("Invalid input. Please enter a positive integer.")
        break

torch.manual_seed(42)

train_essays_df = pd.read_csv('Datasets/AI_Human.csv')
#print(train_essays_df.info())

n_rows = train_essays_df['generated'].nunique()
#print(n_rows)

train_texts = train_essays_df['text'].tolist()
train_labels = train_essays_df['generated'].tolist()

tokenized_corpus = [tokenize(text) for text in train_texts]
combined_corpus = []

for text in tokenized_corpus:
    for token in text:
        combined_corpus.append(token)
word_freqs = Counter(combined_corpus)

Max_Vocab_Size = 5000
most_common_words = word_freqs.most_common(Max_Vocab_Size)

print("-----------")
#print("Top 15 most common words:", most_common_words[0:15])
#print('-----------')

vocab = {word: idx+2 for idx, (word, _) in enumerate(most_common_words)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

Max_Seq_Length = 512
padded_text_seqs = [pad_or_truncate(encode_text(text, vocab), max_len=Max_Seq_Length) for text in train_texts]

X_tensor_train, X_tensor_test, y_tensor_train, y_tensor_test = train_test_split(
    torch.tensor(padded_text_seqs), torch.tensor(train_labels, dtype=torch.long), train_size=.8, test_size=0.2, random_state=42
)

batch_size=64
train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_tensor_test, y_tensor_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

vocab_size = len(vocab)
embed_size = 800
hidden_size = 2048
hidden_size2 = 1024
hidden_size3 = 512
hidden_size4 = 256
output_size = n_rows

model = None

if model_type == "N":
    model = SimpleNNWithEmbeddingN(vocab_size=len(vocab), output_size=2)
elif model_type == "L":
    model = SimpleNNWithEmbeddingL(vocab_size=len(vocab), output_size=2)
elif model_type == "XL":
    model = SimpleNNWithEmbeddingXL(vocab_size=len(vocab), embed_size=800, hidden_size=2048, hidden_size2=1024, hidden_size3=512, hidden_size4=256, output_size=2)

if train == True:
    train_model(model, train_dataloader, num_epochs=epochs)
    torch.save(model, save_path)
else:
    model = torch.load(save_path, weights_only=False)

confusion_report(model, test_dataloader, model_type)

human_text = '“... set!” he shouts. Half a second later, he presses the trigger. The hammer hits the powder, and a flame is ignited. A millisecond later, the receiver picks up the sound. It sends a signal through the air at the speed of light, and the timer receives it and starts the clock. At the same time, the sound reaches my eardrum, causing my eardrum to vibrate. This is amplified by the bones in my ear, causing the fluid in my cochlea to bend the hair cells. This sends an electrical signal to my brain, and it recognizes the sound. The moment I have been waiting for has occurred. I take one step, two steps, four steps, six steps. By now, the crowd is forming a wall in front of me. “Now,” I think to myself, “the race really starts. I want to pass this group on the outside of the turn, but I know theyre going too fast for me. I must pace myself. Two of my teammates fall back towards me, but I’m still behind them. I know without looking that I am last. “I’ll stick with them,” I tell myself, and accelerate to catch them. By now, weve already completed 200 meters. “1400 left, or 12.5% done,” I think. Anything to keep me from thinking about my legs, my breathing, my time … my time! I quickly check my watch.'
print(model_type)
classify_text(model, human_text)
print('-----------')

ai_text = "Coding is like solving a puzzle where you're both the architect and the builder. It starts with an idea, a spark of what you want to create. Then, you break it down into smaller, manageable pieces. Each line of code is a step towards bringing that idea to life. There's a certain rhythm to coding. You write, you test, you debug, and you repeat. It's a cycle of trial and error, of learning and adapting. Sometimes, it's frustrating when things don't work as expected. But when they do, it's incredibly rewarding. Coding is also about collaboration. You share your work, get feedback, and improve. It's a continuous journey of growth and discovery. Whether you're a beginner or an expert, there's always something new to learn."
print(model_type)
classify_text(model, ai_text)