import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import tarfile
import urllib.request

# 1. Baixando recursos NLTK para inglês
nltk.download('punkt')
nltk.download('stopwords')

# 2. Função de pré-processamento

def preprocess_text_en(text):
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    words = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return " ".join(stemmed_words)

# 3. Baixando e carregando o IMDB Movie Reviews

def download_and_extract_imdb():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filename = "aclImdb_v1.tar.gz"
    if not os.path.exists("aclImdb"):
        print("Baixando IMDB Movie Reviews...")
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        print("Extração concluída.")
    else:
        print("IMDB Movie Reviews já está disponível.")

def load_imdb_data(subset="train", n_samples=1000):
    data, labels = [], []
    for label in ["pos", "neg"]:
        path = f"aclImdb/{subset}/{label}"
        files = os.listdir(path)[:n_samples//2]
        for fname in files:
            with open(os.path.join(path, fname), encoding="utf-8") as f:
                text = f.read()
                data.append(preprocess_text_en(text))
                labels.append(1 if label == "pos" else 0)
    return data, labels

# 4. Vetorização com scikit-learn

download_and_extract_imdb()
X, y = load_imdb_data(subset="train", n_samples=1000)
vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# 5. Classificação com scikit-learn (Naive Bayes)
print("\n--- scikit-learn Naive Bayes ---")
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred_nb):.4f}")
print(classification_report(y_test, y_pred_nb, target_names=["Negative", "Positive"]))

# 6. Classificação com PyTorch (rede neural simples)
class IMDBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

train_dataset = IMDBDataset(X_train, y_train)
test_dataset = IMDBDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model_nn = SimpleNN(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

print("\n--- PyTorch Simple NN ---")
for epoch in range(3):
    model_nn.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model_nn(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed.")

# Avaliação
model_nn.eval()
correct, total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model_nn(X_batch)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        all_preds.extend(predicted.tolist())
        all_labels.extend(y_batch.tolist())
print(f"Acurácia: {correct/total:.4f}")
from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

# 7. Exemplo de monitoramento de reputação de marca
print("\n--- Monitorando reputação de marca com frases novas ---")
new_texts = [
    "The product was amazing, I loved it!",
    "Terrible customer service, never buying again.",
    "Fast delivery and great quality.",
    "The app crashes all the time, very frustrating."
]
new_processed = [preprocess_text_en(t) for t in new_texts]
new_vec = vectorizer.transform(new_processed)
# scikit-learn
preds_nb = model_nb.predict(new_vec)
for text, pred in zip(new_texts, preds_nb):
    print(f"[Naive Bayes] '{text}' -> Reputation: {'Positive' if pred==1 else 'Negative'}")
# PyTorch
model_nn.eval()
with torch.no_grad():
    preds_nn = model_nn(torch.tensor(new_vec.toarray(), dtype=torch.float32))
    preds_nn = torch.argmax(preds_nn, dim=1).tolist()
for text, pred in zip(new_texts, preds_nn):
    print(f"[PyTorch NN] '{text}' -> Reputation: {'Positive' if pred==1 else 'Negative'}")
