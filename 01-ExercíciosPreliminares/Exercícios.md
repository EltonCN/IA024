
https://docs.google.com/document/d/1fjSqjM0mBkSIJrqFZJLfkoElMF13yl0log8hdwX339c/edit?pli=1
https://colab.research.google.com/drive/1fcAZPcj-0XfFU0BmkcPdC7xwl-LbY2gG#scrollTo=APN8TsSrVXwB


Leia o artigo: On the Opportunities and Risks of Foundation Models (Até a seção 1, página 12) e faça uma lista de tópicos que você achou importante. Em particular, aponte o que mudou desde a escrita do artigo em julho de 2022.

Coloque o Link do arquivo PDF com a lista de tópicos e a sua resposta. (não exceder uma página)
https://arxiv.org/pdf/2108.07258.pdf


Elton Cardoso do Nascimento
233840

IA024 - Redes Neurais Profundas para Processamento de Linguagem Natural - 1s2024

# I - Vocabulário e tokenização

## Exercícios I.1

Na célula de calcular o vocabulário, aproveite o laço sobre IMDB de treinamento e utilize um segundo contador para calcular o número de amostras positivas e amostras negativas. Calcule também o comprimento médio do texto em número de palavras dos textos das amostras.


**Respostas:**

### I.1.a) a modificação do trecho de código

```python
# limit the vocabulary size to 20000 most frequent tokens
vocab_size = 20000

counter = Counter()
counterTarget = Counter() # NEW: counter for target
meanN_Word = 0 # NEW: mean words in a sentence
for (target, line) in list(IMDB(split='train')):
    words = line.split() # NEW: avoid duplication
    counter.update(words) # CHANGED: use words variable

    # NEW: Update target counter
    if target == 1: 
      counterTarget.update("-")
    else:
      counterTarget.update("+")

    meanN_Word += len(words) # NEW: add the sentence lenght

nTrain = len(list(IMDB(split='train'))) # NEW: store size of train dataset
meanN_Word /= nTrain # NEW: compute mean words per sentence

# create a vocabulary of the 20000 most frequent tokens
most_frequent_words = sorted(counter, key=counter.get, reverse=True)[:vocab_size]
vocab = {word: i for i, word in enumerate(most_frequent_words, 1)}
vocab_size = len(vocab)

# NEW: print required results
print("Amostras positivas:", counterTarget["+"])
print("Amostras negativas:", counterTarget["-"])
print("Total de amostras:", nTrain)
print("Comprimento médio:", meanN_Word, "palavras")
```


### I.1.b) número de amostras positivas, amostras negativas e amostras totais

Amostras positivas: 12500\
Amostras negativas: 12500\
Total de amostras: 25000

### I.1.c) comprimento médio dos textos das amostras (em número de palavras)

Comprimento médio: 233.7872 palavras

## Exercícios I.2

As linhas 9 e 10 da célula do vocabulário são linhas típicas de programação python em listas com dicionários com laços na forma compreensão de listas ou list comprehension em inglês. Procure analisar e estudar profundamente o uso de lista e dicionário do python. Estude também a função encode_sentence.

Mostre as cinco palavras mais frequentes do vocabulário e as cinco palavras menos frequentes. Qual é o código do token que está sendo utilizado quando a palavra não está no vocabulário? Calcule quantos tokens das frases do conjunto de treinamento que não estão no vocabulário.

**Respostas:**
### I.2.a) Cinco palavras mais frequentes, e as cinco menos frequentes. Mostre o código utilizado, usando fatiamento de listas (list slicing).

5 palavras mais frequentes: ['the', 'a', 'and', 'of', 'to']\
5 palavras menos frequentes: ['age-old', 'place!', 'Bros', 'tossing', 'nation,']

Código:

```python
print("5 palavras mais frequentes:", most_frequent_words[:5])
print("5 palavras menos frequentes:", most_frequent_words[-5:])
```


### I.2.b) Explique onde está a codificação que atribui o código de "unknown token" e qual é esse código.

Na linha 

```python    
return [vocab.get(word, 0) for word in sentence.split()] # 0 for OOV
```

O método `get(word, 0)` retornará "0" caso a palavra contida em `word` não esteja no vocabulário, visto que o valor padrão caso o elemento não esteja no dicionário foi definido para "0" (segundo parâmetro do método). Sendo "0" então o código para um token desconhecido.


### I.2.c) Calcule o número de unknown tokens no conjunto de treinamento e mostre o código de como ele foi calculado.

DÚVIDA: A questão está perguntando quantas palavras em todas as sentenças não estão no vocabulário, ou quantas vezes aparece uma palavra em uma sentença que não está no vocabulário? (Ou seja, se uma mesma palavra desconhecida aparecer mais de uma vez, eu devo computar apenas a primeira vez que ela aparece ou todas as vezes que aparece?)


Total de tokens desconhecidos: 566141

Código utilizado:

```python
def count_unknow(sentence):
  n_unknown = 0
  encoded_sentence = encode_sentence(sentence, vocab)
  for encoded_word in encoded_sentence:
    if encoded_word == 0:
      n_unknown += 1

  return n_unknown

total_unknown = 0

for (target, line) in list(IMDB(split='train')):
  encode_sentence(line, vocab)

  total_unknown += count_unknow(line)

print("Total de tokens desconhecidos:", total_unknown)
```

## Exercícios I.3:

**Reduzindo o número de amostras para 200**

Uma forma simples de reduzir o número de amostras é utilizar o fatiamento de listas para selecionar apenas as primeiras 200 amostras utilizando [:200] na lista do IMDB: list(IMDB(split='train'))[:200].
Faça isto, tanto na linha 5 da célula de calcular o vocabulário como na linha 5 da célula do "II - Dataset". 
Com estas duas modificações, execute o notebook por completo novamente. Você verá que o tempo de processamento cairá drasticamente, para aproximadamente 1 a 2 segundos por época. Porém você vai notar que a Acurácia calculada na célula VI - Avaliação sobe para 100% ou próximo disso.
Consegue justificar a razão deste resultado inesperado, entendendo que no treinamento, as perdas em cada época continuam próximas de valores com todo o dataset?
Para ver a resposta, verifique agora no dataset com 200 amostras, quantas são as amostras positivas e quantas são as amostras negativas no dataset de teste.


**Respostas:**
### I.3.a) Qual é a razão pela qual o modelo preditivo conseguiu acertar 100% das amostras de teste do dataset selecionado com apenas as primeiras 200 amostras?

Analisando a nova distribuição de amostras:

- Amostras positivas: 0
- Amostras negativas: 200

é possível observar que o subconjunto de amostras de treino selecionado é enviesado, o modelo precisa apenas aprender a predizer "negativo" na saída, simplificando ao extremo a função que precisa ser aprendida.


### I.3.b) Modifique a forma de selecionar 200 amostras do dataset, porém garantindo que ele continue balanceado, isto é, aproximadamente 100 amostras positivas e 100 amostras negativas.

Com o novo código para selecionar o subconjunto:

```python
trainDataset = []
counterTarget = Counter()
nTrain = 0
for (target, line) in list(IMDB(split='train')):

  if target == 1 and counterTarget["-"] < 100: 
    trainDataset.append((target, line))

    counterTarget.update("-")
    nTrain += 1
  elif target == 2 and counterTarget["+"] < 100:
    trainDataset.append((target, line))
    
    counterTarget.update("+")
    nTrain += 1

  if nTrain == 200:
    break
```

Temos que o resultado se torna não enviesado:

Amostras positivas: 100\
Amostras negativas: 100\
Total de amostras: 200\
Comprimento médio: 230.735 palavras

E a acurácia volta a obter um valor próximo do original (51,744%):

Test Accuracy: 49.0%


# II - Dataset

Precisamos entender como funciona a classe IMDBDataset. Ela é a classe responsável para acessar cada amostra do dataset.

Em primeiro lugar precisamos entender qual será a entrada da rede neural para decidir se o texto é uma crítica positiva ou negativa. Uma das formas mais simples de construir um modelo preditivo é com base nas palavras utilizadas no texto. A distribuição das palavras de um texto tem alta correlação com o fato do texto estar falando bem ou falando mal de um filme. Certamente é estimativa que possui seus erros, mas é a forma mais simples e eficiente de se fazer uma análise de sentimento ou de maneira geral uma classificação de um texto. Esse método é denominado "Bag of Words". A entrada da rede neural, para cada amostra, será um vetor de comprimento do vocabulário, com valores todos zero, com exceção dos tokens que aparecem no texto da amostra. Esse método de codificação é também denominado "One-Hot". Estude o código da classe IMDBDataset fazendo experimentos e perguntas ao chatGPT para entender com profundidade esta classe.

## Exercício II.1:

### II.1.a) Investigue o dataset criado na linha 24. Faça um código que aplique um laço sobre o dataset train_data e calcule novamente quantas amostras positivas e negativas do dataset.

DÚVIDA: preciso calcular as estatísticas para o dataset de treino original, ou o reduzido não enviesado criado na questão I.3?

O dataset criado na linha 24


### II.1.b) Calcule também o número médio de palavras codificadas em cada vetor one-hot. Compare este valor com o comprimento médio de cada texto (contado em palavras), conforme calculado no exercício 

O número médio é X, o que é menor que o comprimento médio Y.

### I.1.c) e explique a diferença.

Essa diferença é devido ao fato de que, se uma palavra aparecer várias vezes no texto, o primeiro método irá contar cada aparição, enquanto que olhar o tamanho médio do vetor one-hot considera apenas 1 aparição de cada palavra.

---

**Aumentando a eficiência do treinamento com o uso da GPU T4**

O código do notebook está preparado para executar tanto com ambiente usando CPU como com GPU, entretanto o ganho de velocidade está sendo reduzido de 45 segundos para 29 segundos que é um ganho muito aquém do esperado que seria ter um speedup entre 7 e 11 vezes dependendo da aplicação. Vamos entender a razão desta baixa eficiência e corrigir o problema.

A GPU é utilizada durante o treinamento do modelo, onde é utilizada a técnica de minimização da Loss utilizando o gradiente descendente. Isso ocorre na segunda célula do "V - Laço de Treinamento". Iremos analisar os detalhes mais à frente, para por enquanto basta entender onde a GPU é utilizada. A linha 17 é onde o modelo está fazendo a predição (passo forward), dado a entrada, calcula a saída da rede (muitas vezes chamado de logito) e o cálculo da loss está sendo feito na linha seguinte e o cálculo do gradiente ocorre na linha 21 e a linha 22 é onde ocorre o ajuste dos parâmetros (weights) da rede neural fazendo ela minimizar a Loss. Esse é o processo que é mais demorado e onde a GPU tem muitos ganhos, pois envolve praticamente apenas multiplicação de matrizes. Existem apenas 3 linhas que controlam o uso da GPU que servem para colocar o modelo, a entrada e a saída esperada (targets) na GPU: linhas 3, 14 e 15, respectivamente.


---