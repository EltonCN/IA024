
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

A rede neural será alimentada pelo vetor one-hot (quais suas dimensões) e fará uma predição da probabilidade do texto associado ao one-hot ser uma mensagem positiva.

**Aumentando a eficiência do treinamento com o uso da GPU T4**

O código do notebook está preparado para executar tanto com ambiente usando CPU como com GPU, entretanto o ganho de velocidade está sendo reduzido de 45 segundos para 29 segundos que é um ganho muito aquém do esperado que seria ter um speedup entre 7 e 11 vezes dependendo da aplicação. Vamos entender a razão desta baixa eficiência e corrigir o problema.

A GPU é utilizada durante o treinamento do modelo, onde é utilizada a técnica de minimização da Loss utilizando o gradiente descendente. Isso ocorre na segunda célula do "V - Laço de Treinamento". Iremos analisar os detalhes mais à frente, para por enquanto basta entender onde a GPU é utilizada. A linha 17 é onde o modelo está fazendo a predição (passo forward), dado a entrada, calcula a saída da rede (muitas vezes chamado de logito) e o cálculo da loss está sendo feito na linha seguinte e o cálculo do gradiente ocorre na linha 21 e a linha 22 é onde ocorre o ajuste dos parâmetros (weights) da rede neural fazendo ela minimizar a Loss. Esse é o processo que é mais demorado e onde a GPU tem muitos ganhos, pois envolve praticamente apenas multiplicação de matrizes. Existem apenas 3 linhas que controlam o uso da GPU que servem para colocar o modelo, a entrada e a saída esperada (targets) na GPU: linhas 3, 14 e 15, respectivamente.


---

## Exercício II.2:

Com a o notebook configurado para GPU T4, meça o tempo de dois laços dentro do for da linha 13 (coloque um break após dois laços) e determine quanto demora demora para o passo de forward (linhas 14 a 18), para o backward (linhas 20, 21 e 22) e o tempo total de um laço. Faça as contas e identifique o trecho que é mais demorado. 

### II.2.a) 

|Momento| Tempo
-|-
Tempo do laço:| 128.39257717132568 ms
Tempo do forward:| 1.8529891967773438 ms
Tempo do backward:| 1.2031793594360352 ms

(o tempo indicado é a média das duas execuções solicitadas)

Conclusão: A demora principal está na execução da linha: 13 = `for inputs, targets in train_loader:`. Isso pode ser ainda mais explicitado calculando o tempo médio que demora para executá-la, 125.31495094299316 ms, que é praticamente o tempo inteiro do laço.

### II.2.b) Trecho que precisa ser otimizado. (Esse é um problema bem mais difícil) A dica aqui é que precisa muito de conceitos de iterador e programação orientada a objetos.

O trecho que precisa ser otimizado é o método `__getitem__` da classe `IMDBDataset`, visto que relizar o encoding de cada amostra sempre que é utilizada é custoso e está gerando o aumento no tempo.

### II.2.c) Otimize o código e explique aqui.

Uma solução simples para acelerar o código é realizar um cacheamento dos dados: todos os encodings são calculados anteriormente, e todos os targets são convertidos para tensores, e durante o treino é realizado apenas um acesso:

```python
class IMDBDataset(Dataset):
    def __init__(self, split, vocab):
        #if split == "train":
        #
        #  self.data = trainDataset
        #else:
        #  self.data = list(IMDB(split=split))
        self.data = list(IMDB(split=split))

        self.vocab = vocab

        #####################
        ##HERE
        #####################
        
        # Cache data encoding and target
        self.itens = []
        for idx in range(len(self.data)):
          target, line = self.data[idx]
          target = 1 if target == 1 else 0

          # one-hot encoding
          X = torch.zeros(len(self.vocab) + 1)
          for word in encode_sentence(line, self.vocab):
              X[word] = 1

          self.itens.append((X, torch.tensor(target)))
          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        #####################
        ##AND HERE
        #####################


        #Only retrieves data
        return self.itens[idx]
```

Essa alteração diminui os tempo do laço de treino, gerando um speedup de ~20x:

|Momento| Tempo anterior (ms) | Novo tempo (ms) | Speedup
-|-|-|-
Tempo do laço:| 128.39257717132568 | 6.272673606872559 | 20.468556985119445
Tempo do forward:| 1.8529891967773438 | 1.9022226333618164 | ~ 1
Tempo do backward:| 1.2031793594360352 | 1.0069608688354492 | ~ 1
Tempo de acessar o dado: | 125.31495094299316 | 3.3473968505859375 | 37.43653846153846


Como um dado é acessado várias vezes durante todo o processo de treino, isso irá gerar uma melhora na performance global (não apenas uma melhora no processo de treino, o que aconteceria se ele fosse acessado apenas 1 vez). 


Para um dataset maior, é possível também paralelizar o cálculo dos encodings, que é custoso, visto que o tempo para criar o objeto do dataset aumentou para 26.34220790863037 s.

---

Após esta otimização, **é esperado que o tempo de processamento de cada época caia tanto para execução em CPU (da ordem de 10 segundos por época) como para GPU T4 (da ordem de 1 a 2 segundos por época)**. Isso utilizando as 25 mil amostras do dataset IMDB inteiro.

Atenção: Se não conseguir atingir esse objetivo, procure árduamente entender com maiores detalhes o código. Esse exercício é fundamental para poder acompanhar o curso durante o semestre.

Agora que a execução está bem mais otimizada em tempos de execução, modifique o início do notebook para ter novamente o dataset completo: 25 mil amostras e vamos analisar um outro fator importante que é a escolha do LR (Learning Rate)

---

**Escolhendo um bom valor de LR**

## Exercício II.3:

Faça a melhor escolha do LR, analisando o valor da acurácia no conjunto de teste, utilizando para cada valor de LR, a acurácia obtida. Faça um gráfico de Acurácia vs LR e escolha o LR que forneça a maior acurácia possível. Atenção, mantenha o número de épocas como 5.

### II.3.a) Gráfico Acurácia vs LR

Calculando a acurácia média em 5 treinos variando linearmente a taxa de treino entre 1E-6 e 1 com 10 elementos, e adicionando o valor "2" para verificar a influência de valores muito grandes, temos que a acurácia varia da seguinte forma:

![](ImpactoDaTaxaDeTreino.png)


### II.3.b) Valor ótimo do LR (para isso é desejável que o LR ótimo que forneça maior acurácia no conjunto de testes seja maior que usar um LR menor e um LR maior que o LR ótimo.)

Observando os valores obtidos, temos que a acurácia é máxima para LR = 0.111112

### II.3.c) Mostre a equação utilizada no gradiente descendente e qual é o papel do LR no ajuste dos parâmetros (weights) do modelo da rede neural.

