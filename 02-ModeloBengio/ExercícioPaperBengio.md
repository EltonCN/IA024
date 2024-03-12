Elton Cardoso do Nascimento
233840

IA024 - Redes Neurais Profundas para Processamento de Linguagem Natural - 1s2024

# Leitura do Artigo " A neural probabilistic language model. Journal of Machine Learning Research."

A principal contribuição do artigo é a proposta de um novo modelo de linguagem capaz de superar os modelos estado da arte da época. Isso foi possível devido a uma arquitetura que tenta capturar a similaridade de palavras junto da probabilidade de da próxima palavra. Ela utiliza representações contínuas aprendidas das palavras, o que permite uma maior generalização e um uso mais inteligente dos dados de treino: palavras similares geram embeddings similares, o que faz com que uma determinada sentença no conjunto permita também o aprendizado de sentenças com palavras similares. Isso permitiu a arquitetura evitar o problema de dimensionalidade ao aprendender várias sentenças através de uma. Apresenta também detalhes da implementação e resultados experimentais do novo modelo.

Mais além do trabalho realizado, ele já apresenta diversas formas de melhorar a arquitetura proposta, como o uso de embedding também na palavra de saída.