# Tema 06: Processamento de Linguagem Natural

## Descrição
Este projeto tem como objetivo classificar um pedaço de texto (análise de sentimentos) utilizando técnicas de Processamento de Linguagem Natural (PLN).

- **Tema escolhido:** Classificação de texto (sentimento positivo/negativo)
- **Exemplo de aplicação:** Monitorar a reputação de uma marca qualquer a partir de opiniões de clientes.

## Bibliotecas Utilizadas
- NLTK
- scikit-learn
- (Sugestão para expansão: PyTorch)

## Repositórios de Dados
- IMDB Movie Reviews (sugestão para datasets maiores)
- Neste projeto, foi utilizado um dataset manual simples para fins didáticos.

## Como Executar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o script principal:
   ```bash
   python main.py
   ```

## Observações
- O código está preparado para trabalhar com textos em inglês.
- O dataset pode ser expandido facilmente para melhorar a performance do modelo.
- Para aplicações reais, recomenda-se utilizar um dataset maior, como o IMDB Movie Reviews.

## Exemplo de Teste Utilizado no Desenvolvimento

Durante o desenvolvimento, foi utilizado o seguinte conjunto de opiniões para testar a reputação da marca Acme:

```python
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
```

Esse teste foi inserido para avaliar a capacidade do modelo em classificar opiniões variadas, tanto positivas quanto negativas, simulando cenários reais de reputação de marca.

---

Projeto desenvolvido para estudo de PLN e classificação de sentimentos.
