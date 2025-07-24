import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer # Stemmer para inglês

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report

from collections import Counter



# Baixando os recursos necessários do NLTK para inglês

nltk.download('punkt')

nltk.download('stopwords')

# O recurso 'rslp' é específico para português e não é utilizado neste projeto em inglês
# nltk.download('rslp')



# --- 1. Pré-processamento do Texto em Inglês ---

def preprocess_text_en(text):

    """
    Função para limpar e normalizar o texto em inglês:
    - Tokenização: Divide o texto em palavras (tokens).
    - Remoção de Stopwords: Remove palavras comuns do inglês (ex: 'and', 'or', 'the').
    - Stemming: Reduz as palavras ao seu radical (em inglês).
    - Converte para minúsculas e remove caracteres não alfabéticos.
    """
    # Tokenização simples usando regex (compatível com inglês)
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())



    # Remoção de caracteres não alfabéticos

    words = [word for word in tokens if word.isalpha()]



    # Remoção de stopwords em inglês

    stop_words = set(stopwords.words('english'))

    filtered_words = [word for word in words if word not in stop_words]



    # Stemming (Radicalização em inglês)

    stemmer = PorterStemmer()

    stemmed_words = [stemmer.stem(word) for word in filtered_words]



    return " ".join(stemmed_words)



# --- 2. Carregamento e Preparação dos Dados (Opiniões sobre a marca Acme) ---

print("Creating and loading customer opinions about the brand Acme...")

texts = [
    "Acme's customer service was outstanding, they solved my issue quickly!",
    "I'm very disappointed with Acme, the product broke after one week.",
    "I love Acme's new smartphone, it's fast and reliable.",
    "Acme never responds to my emails, terrible support.",
    "The quality of Acme's products keeps getting better.",
    "I had a bad experience with Acme's delivery, it was late and the box was damaged.",
    "Acme exceeded my expectations, highly recommend!",
    "The Acme app is confusing and crashes often.",
    "Acme's staff was friendly and helpful at the store.",
    "I regret buying from Acme, the warranty process is a nightmare.",
    "Acme's prices are very affordable compared to competitors.",
    "The Acme website is slow and hard to navigate.",
    "Acme always delivers on time, I'm satisfied.",
    "I had to return my Acme product, the process was easy and fast.",
    "Acme's packaging is eco-friendly, I appreciate that.",
    "The Acme hotline kept me on hold for over an hour.",
    "Acme's new update made the app much better.",
    "I received the wrong item from Acme, very frustrating.",
    "Acme's loyalty program offers great discounts.",
    "The Acme store was clean and well organized."
]

# Labels: 1 for positive, 0 for negative

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]

categories = ['Negative', 'Positive']



# Aplicando o pré-processamento aos textos

print("Applying preprocessing to customer opinions...")

X_processed = [preprocess_text_en(text) for text in texts]



# --- 3. Extração de Características (Vetorização) ---

print("Converting text to vectors with TF-IDF...")

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X_processed)



# --- 4. Divisão do Dataset em Treino e Teste ---

# Com um dataset pequeno, podemos usar uma divisão simples ou até mesmo treinar com tudo.

# Para manter a estrutura, vamos dividir.

# Garantir balanceamento

print(f"Treinando com {len(labels)} exemplos. Distribuição: {Counter(labels)}")

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.3, random_state=42, stratify=labels)

print(f"Dataset dividido: {X_train.shape[0]} treino e {X_test.shape[0]} teste.")



# --- 5. Treinamento do Modelo de Classificação ---

print("Treinando o modelo Naive Bayes apenas com os dados de treino...")

model = MultinomialNB()

model.fit(X_train, y_train)



# --- 6. Avaliação do Modelo ---

print("Avaliando o modelo nos dados de teste...")

y_pred_test = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)

report = classification_report(y_test, y_pred_test, target_names=categories, zero_division=0)



print(f"\nAcurácia nos dados de teste: {accuracy:.4f}")

print("\nRelatório de Classificação nos dados de teste:")

print(report)



# --- 7. Testando o Modelo com Novas Opiniões ---

print("\n--- Testing the model with new customer opinions ---")

new_texts = [
    # Positivas
    "Acme's support team was very helpful and polite.",
    "Acme delivered my order on time and in perfect condition.",
    "Acme's website is user-friendly and easy to navigate.",
    "I love the discounts I get as an Acme member.",
    "Acme always surprises me with their fast delivery.",
    "Acme's new product line is innovative and high quality.",
    "I received excellent assistance at the Acme store.",
    "The Acme app update fixed all previous bugs.",
    "Acme sent me the wrong color, but quickly fixed the issue.",
    # Negativas
    "I will never buy from Acme again, worst experience ever.",
    "The Acme product stopped working after a few days.",
    "Acme's customer service ignored my complaint.",
    "The packaging from Acme was damaged when it arrived.",
    "I had to wait too long for a response from Acme support.",
    "Acme's return policy is confusing and unhelpful."
]



# Pré-processamento e vetorização das novas frases

processed_new_texts = [preprocess_text_en(text) for text in new_texts]

new_texts_tfidf = vectorizer.transform(processed_new_texts)



# Realizando a predição

predictions = model.predict(new_texts_tfidf)



for text, pred_label in zip(new_texts, predictions):

    print(f"Opinion: '{text}' -> Reputation: '{categories[pred_label]}'")