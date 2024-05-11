Elton Cardoso do Nascimento - 233840

IA024 - Redes Neurais Profundas para Processamento de Linguagem Natural - 1s2024

# Leitura do Artigo "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al.)

O artigo propõe a técnica ReAct, que se baseia em realizar um projeto conjunto com uma LLM de raciocínio e ação. A etapa de raciocínio se assemelha a técnica CoT (chain-of-thought), criando um parágrafo explicando um raciocínio relacionado ao problema; enquanto que a etapa de ação envolve o uso de diferentes ações disponíveis ao agente, permitindo-o interagir e obter observações do mundo externo. O trabalho formula a técnica em um problema de aprendizado por reforço, em que a política $\pi(a_t|c_t)$ depende do contexto $c$ de ações e observações anteriores, e as ações podem consistir tanto de ações que afetam o ambiente externo ou ações "pensamento", que apenas adicionam informação útil ao contexto através do próprio contexto e informações internalizadas no modelo. Devido a dificuldade de aprender tal conjunto de ações, um LLM congelada com few-shot prompting é utilizado, permitindo aprender uma política com poucos dados de exemplo.

