# 🧮 MathSolver-QLoRA: Fine-tuning Eficiente de LLMs para Resolução de Problemas Matemáticos de Nível Competitivo

> **Um framework de fine-tuning com QLoRA 4-bit para adaptar Large Language Models à resolução de problemas matemáticos de alta complexidade, incluindo provas olímpicas e vestibulares brasileiros, com 97% de acurácia em GPU consumer-grade (8GB VRAM).**

<p align="center">

[![Benchmark](https://img.shields.io/badge/Benchmark_Accuracy-97%25-brightgreen?style=for-the-badge)](outputs/eval/benchmark/)
[![Base Model](https://img.shields.io/badge/Base_Model-Qwen2.5--Math--7B-blue?style=for-the-badge)](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
[![Quantization](https://img.shields.io/badge/Quantization-4bit_NF4-purple?style=for-the-badge)](#metodologia)
[![VRAM](https://img.shields.io/badge/VRAM_Required-5.5GB-orange?style=for-the-badge)](#requisitos)

</p>

---

## 📋 Índice

- [Resumo](#resumo)
- [Motivação](#motivação)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Demonstrações](#demonstrações)
- [Reprodutibilidade](#reprodutibilidade)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Limitações e Trabalhos Futuros](#limitações-e-trabalhos-futuros)
- [Referências](#referências)

---

## Resumo

Este projeto demonstra que é possível **adaptar um LLM de 7B parâmetros** para resolver problemas matemáticos de nível olímpico e vestibular (ITA, OBMEP, Unicamp, Fuvest, IME) utilizando:

- **QLoRA (Quantized Low-Rank Adaptation)** com quantização NF4 de 4 bits
- **Apenas 126 exemplos curados** de alta qualidade em português
- **Uma GPU consumer-grade** (NVIDIA RTX 3070, 8GB VRAM)
- **~90 minutos de treino**

O modelo resultante atinge **97% de acurácia** (29/30) em um benchmark de 30 questões de alta complexidade, com resolução passo a passo e raciocínio explícito.

### Contribuições Principais

1. **Eficiência de dados**: demonstramos que 126 exemplos curados são suficientes para fine-tuning eficaz quando combinados com um modelo base forte
2. **Acessibilidade**: o pipeline inteiro roda em hardware consumer-grade (8GB VRAM)
3. **Raciocínio explícito**: o modelo produz chain-of-thought detalhado em português
4. **Framework reprodutível**: código aberto para treino, benchmark e demonstração

---

## Motivação

A resolução automatizada de problemas matemáticos é um dos desafios centrais da IA. Enquanto modelos como GPT-4 e Gemini demonstram capacidades impressionantes, eles:

- Requerem infraestrutura cara (cloud APIs)
- Não são especializados para o contexto educacional brasileiro
- Não permitem customização local

Este projeto aborda essas limitações ao criar um **modelo local, especializado e eficiente** que resolve problemas de vestibulares e olimpíadas brasileiras com alta acurácia.

---

## Metodologia

### Modelo Base

**Qwen2.5-Math-7B-Instruct** — um LLM de 7.7 bilhões de parâmetros pré-treinado especificamente para raciocínio matemático pela equipe Qwen/Alibaba.

### Quantização e Adaptação

```
┌─────────────────────────────────────────────────────┐
│           Qwen2.5-Math-7B-Instruct                  │
│           7.7B parâmetros (congelados)               │
│           Quantização: NF4 4-bit double-quant        │
│                                                     │
│    ┌─────────────────────────────────────┐           │
│    │     QLoRA Adapter (treinável)       │           │
│    │     rank = 32, alpha = 64           │           │
│    │     80M parâmetros (1.05%)          │           │
│    │     Módulos: q,k,v,o,gate,up,down   │           │
│    └─────────────────────────────────────┘           │
│                                                     │
│    VRAM total: 5.50 GB (de 8 GB disponíveis)        │
└─────────────────────────────────────────────────────┘
```

### Pipeline de Treino

| Etapa | Descrição |
|-------|-----------|
| **1. Curadoria** | 126 problemas selecionados de ENEM, OBMEP, ITA, Unicamp com soluções detalhadas passo a passo |
| **2. Formatação** | Cada exemplo formatado como chat (system → user → assistant) com padrão "Final answer: X" |
| **3. Quantização** | Modelo base carregado em 4-bit NF4 com double quantization |
| **4. LoRA Config** | Adaptadores r=32, α=64 inseridos em todas as camadas de atenção + FFN |
| **5. Treino** | 5 epochs, batch efetivo=4, lr=1.5e-4, cosine scheduler, gradient checkpointing |
| **6. Avaliação** | Benchmark de 30 questões (3 níveis × 13 categorias) |

### Hyperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| LoRA rank (r) | 32 | Balance entre capacidade e eficiência |
| LoRA alpha (α) | 64 | Escala 2× para compensar rank médio |
| Learning rate | 1.5 × 10⁻⁴ | Padrão para QLoRA com AdamW 8-bit |
| Epochs | 5 | Convergência sem overfitting para 126 exemplos |
| Max sequence length | 1536 tokens | Acomoda soluções longas com raciocínio detalhado |
| Gradient accumulation | 4 | Batch efetivo = 4 (limite de VRAM) |
| Optimizer | Paged AdamW 8-bit | Reduz uso de memória |
| Quantization | NF4 + double quant | Máxima compressão com preservação de qualidade |

### Curva de Treino

```
Loss:     ████████████████████░  0.497  (epoch 1)
          ██████████░░░░░░░░░░  0.165  (epoch 2)
          ████░░░░░░░░░░░░░░░░  0.078  (epoch 3)
          ███░░░░░░░░░░░░░░░░░  0.056  (epoch 4)
          ██░░░░░░░░░░░░░░░░░░  0.049  (epoch 5)

Accuracy: ████████████████████░  89.8%  (epoch 1)
          ██████████████████░░  96.6%  (epoch 3)
          ███████████████████░  98.8%  (epoch 5)
```

---

## Resultados

### Benchmark Principal: 30 Questões de Alta Complexidade

| Nível de Dificuldade | Questões | Acertos | Acurácia |
|---------------------|----------|---------|----------|
| Médio (ENEM) | 1 | 1 | **100%** |
| Alto (Vestibular Difícil) | 18 | 17 | **94%** |
| Olímpico (OBMEP Fase 2/3) | 11 | 11 | **100%** |
| **TOTAL** | **30** | **29** | **97%** |

### Acurácia por Categoria Matemática

| Categoria | Acertos | Taxa |
|-----------|---------|------|
| Álgebra | 5/5 | 100% ✅ |
| Geometria | 5/5 | 100% ✅ |
| Funções | 3/3 | 100% ✅ |
| Combinatória | 3/3 | 100% ✅ |
| Probabilidade | 2/2 | 100% ✅ |
| Teoria dos Números | 2/2 | 100% ✅ |
| PA/PG (Sequências) | 2/2 | 100% ✅ |
| Geometria Analítica | 2/2 | 100% ✅ |
| Logaritmos | 1/1 | 100% ✅ |
| Matrizes | 1/1 | 100% ✅ |
| Estatística | 1/1 | 100% ✅ |
| Financeira | 1/1 | 100% ✅ |
| Trigonometria | 1/2 | 50% ⚠️ |

### Evolução Comparativa

| Configuração | Dataset | Benchmark | Acurácia |
|-------------|---------|-----------|----------|
| Qwen2.5-Math-1.5B (baseline) | — | 15q básicas | 60% |
| Qwen2.5-Math-7B + QLoRA v1 | 51 exemplos | 13q básicas | 92% |
| Qwen2.5-Math-7B + QLoRA v1 | 51 exemplos | 30q avançadas | 83% |
| **Qwen2.5-Math-7B + QLoRA v2** | **126 exemplos** | **30q avançadas** | **97%** |

> **Ganho absoluto de +37 pontos percentuais** sobre o baseline (60% → 97%).

### Showcase: Problemas Nível ITA/IME (14/15 corretos)

Testado em 15 problemas genuinamente complexos de ITA, IME, OBMEP Fase 3, Unicamp e Fuvest:

| # | Origem | Problema | Resultado |
|---|--------|----------|-----------|
| 1 | ITA | x³+y³ dado x+y=10, xy=21 → **370** | ✅ |
| 2 | OBMEP F3 | 7²⁰²³ mod 11 → **2** | ✅ |
| 3 | Unicamp | log₂x+log₄x+log₈x=11 → **x=64** | ✅ |
| 4 | ITA | \|(2+3i)²+(1-i)³\| → **√149** | ✅ |
| 5 | OBMEP F3 | Coloração hexágono 3 cores | ❌ |
| 6 | Fuvest | f(f(x))=3, f quadrática → **4 raízes** | ✅ |
| 7 | ITA | \|x²-5x+4\|<2 → **intervalos corretos** | ✅ |
| 8 | OBMEP F3 | Lei dos cossenos → **BC=7** | ✅ |
| 9 | Unicamp | aₙ₊₁=2aₙ+1 → **a₁₀=1023 (por indução)** | ✅ |
| 10 | ITA | P(ambas brancas\|≥1 branca) → **2/5** | ✅ |
| 11 | OBMEP F3 | x⁴-4x³+6x²-4x+1 → **(x-1)⁴** | ✅ |
| 12 | Unicamp | Circunferência por 3 pontos → **(2,3), r=√13** | ✅ |
| 13 | IME | max/min de x³-3x²+4 em [-1,3] | ✅ |
| 14 | OBMEP F3 | Último algarismo 3²⁰²⁴+7²⁰²⁴ → **2** | ✅ |
| 15 | ITA | det 3×3 → **1** | ✅ |

---

## Demonstrações

### 🔴 Aritmética Modular (OBMEP Fase 3)
```
PROBLEMA: Encontre o resto da divisão de 7^2023 por 11.

RESOLUÇÃO:
Passo 1: Potências de 7 módulo 11:
  7¹≡7, 7²≡5, 7³≡2, 7⁴≡3, 7⁵≡10,
  7⁶≡4, 7⁷≡6, 7⁸≡9, 7⁹≡8, 7¹⁰≡1.

Passo 2: O ciclo se repete a cada 10 potências.
  2023 = 10×202 + 3.
  7²⁰²³ mod 11 = 7³ mod 11 = 2.

✅ Final answer: 2
```

### 🔴 Álgebra com Identidades (ITA)
```
PROBLEMA: Se x+y = 10 e xy = 21, determine x³+y³.

RESOLUÇÃO:
Passo 1: Identidade: x³+y³ = (x+y)(x²-xy+y²).
Passo 2: x²+y² = (x+y)²-2xy = 100-42 = 58.
Passo 3: x³+y³ = 10×(58-21) = 10×37 = 370.

✅ Final answer: 370
```

### 🔴 Números Complexos (ITA)
```
PROBLEMA: Calcule |z| onde z = (2+3i)² + (1-i)³.

RESOLUÇÃO:
Passo 1: (2+3i)² = 4+12i+9i² = -5+12i.
Passo 2: (1-i)² = -2i → (1-i)³ = (1-i)(-2i) = -2-2i.
Passo 3: z = (-5+12i)+(-2-2i) = -7+10i.
Passo 4: |z| = √(49+100) = √149.

✅ Final answer: √149
```

### 🟡 Recorrência com Prova por Indução (Unicamp)
```
PROBLEMA: a₁=1, a_(n+1)=2aₙ+1. Encontre fórmula fechada e calcule a₁₀.

RESOLUÇÃO:
Passo 1: Termos: a₁=1, a₂=3, a₃=7, a₄=15.
Passo 2: Conjectura: aₙ = 2ⁿ-1.
Passo 3: Prova por indução:
  Base: a₁ = 2¹-1 = 1 ✓
  Passo: a_(n+1) = 2(2ⁿ-1)+1 = 2^(n+1)-1 ✓

✅ Final answer: a₁₀ = 2¹⁰-1 = 1023
```

### 🟡 Geometria Analítica — Sistema 3×3 (Unicamp)
```
PROBLEMA: Circunferência por A(0,0), B(4,0), C(0,6). Centro e raio?

RESOLUÇÃO:
Passo 1: (x-h)² + (y-k)² = r².
Passo 2: A→ h²+k²=r². B→ (4-h)²+k²=r². C→ h²+(6-k)²=r².
Passo 3: B-A: 16-8h=0 → h=2. C-A: 36-12k=0 → k=3.
Passo 4: r² = 4+9 = 13 → r = √13.

✅ Final answer: Centro (2,3), raio √13
```

---

## Reprodutibilidade

### Requisitos

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| GPU | 8 GB VRAM (NVIDIA) | RTX 3070+ |
| RAM | 16 GB | 32 GB |
| CUDA | 12.0+ | 12.8 |
| Python | 3.10+ | 3.10 |
| Disco | 20 GB | 50 GB |

### Instalação

```bash
git clone https://github.com/Brun0Simoes/Modelo_math_Qwen2.5-LoRA.git
cd Modelo_math_Qwen2.5-LoRA

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft bitsandbytes accelerate trl datasets
```

### Treino

```bash
python scripts/train_qlora_7b.py \
    --train-file data/processed/production_dataset_v2.jsonl \
    --output-dir outputs/checkpoints/qwen25math7b_BEST \
    --epochs 5 --batch-size 1 --grad-accum 4 \
    --lr 1.5e-4 --lora-r 32 --lora-alpha 64 --max-seq-len 1536
```

### Inferência

```python
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoPeftModelForCausalLM.from_pretrained(
    "outputs/checkpoints/qwen25math7b_BEST",
    quantization_config=bnb, device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("outputs/checkpoints/qwen25math7b_BEST")
model.eval()

messages = [
    {"role": "system", "content": "Resolva passo a passo."},
    {"role": "user", "content": "Encontre o resto de 7^2023 dividido por 11."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Benchmark

```bash
# Benchmark formal (30 questões, 3 níveis, 13 categorias)
python scripts/benchmark_prova_completa.py

# Showcase demonstrativo (15 problemas ITA/OBMEP/Unicamp)
python scripts/showcase_demo.py
```

---

## Estrutura do Projeto

```
Modelo_math_Qwen2.5-LoRA/
│
├── README.md                          # Este documento
├── .gitignore
│
├── data/processed/                    # Datasets de treino
│   ├── competitive_enem_obmep.jsonl   # 51 problemas ENEM/OBMEP (PT-BR)
│   ├── advanced_training_v2.jsonl     # 75 problemas avançados
│   └── production_dataset_v2.jsonl    # Dataset final combinado (126 ex.)
│
├── scripts/                           # Pipeline completo
│   ├── train_qlora_7b.py             # Treino QLoRA 4-bit
│   ├── benchmark_prova_completa.py   # Avaliação formal (30q)
│   ├── benchmark_7b_direct.py        # Benchmark rápido (13q)
│   ├── showcase_demo.py              # Demo ITA/OBMEP/Unicamp (15q)
│   ├── benchmark_enem_obmep.py       # Benchmark via subprocess
│   ├── build_mega_dataset.py         # Builder de dataset massivo
│   └── solve_problem_json.py        # Solver JSON (integração)
│
├── src/olympiad_system/               # Módulo core
│   └── generator.py                  # Gerador de respostas
│
├── outputs/
│   ├── checkpoints/
│   │   └── qwen25math7b_BEST/        # Adapter de produção
│   │       ├── adapter_config.json   # Configuração LoRA
│   │       ├── training_config.json  # Hyperparâmetros de treino
│   │       └── tokenizer_config.json # Config do tokenizer
│   ├── eval/benchmark/               # Resultados dos benchmarks
│   └── showcase/                     # Resultados do showcase
│
└── apps/math-vision-ui/              # Interface web (Node.js)
    └── server/src/solver.js          # Integração web
```

---

## Limitações e Trabalhos Futuros

### Limitações Atuais

- **Combinatória com grafos**: problemas de coloração (polinômio cromático) requerem mais exemplos de treino
- **Trigonometria**: 1 erro em 2 questões (confusão seno/cosseno)
- **Tempo de inferência**: 40-120s por problema complexo (limitação da GPU 8GB)
- **Idioma do dataset público**: GSM8K e NuminaMath-CoT estão em inglês
- **Figuras**: o modelo não processa imagens/diagramas geométricos

### Trabalhos Futuros

1. **Dataset expandido**: Integrar GSM8K (8.5K) + NuminaMath-CoT (860K) traduzidos para PT-BR
2. **Modelo 14B**: Utilizar GPU com mais VRAM para fine-tuning do Qwen2.5-14B
3. **Multimodal**: Adicionar OCR + visão computacional para processar figuras de provas
4. **Benchmark expandido**: 100+ questões cobrindo todas as provas brasileiras (ITA, IME, Unicamp, FUVEST, UNB, UFMG)
5. **Aplicação web**: Interface completa para estudantes resolverem problemas em tempo real

---

## Referências

1. **Qwen2.5-Math**: Yang, A. et al. (2024). *Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement*. arXiv:2409.12122
2. **QLoRA**: Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized Language Models*. NeurIPS 2023. arXiv:2305.14314
3. **LoRA**: Hu, E.J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
4. **GSM8K**: Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168
5. **NuminaMath**: AI-MO (2024). *NuminaMath-CoT: Chain-of-thought mathematical reasoning dataset*. HuggingFace Hub.

---

## Licença

MIT License. O modelo base (Qwen2.5-Math-7B) está sujeito à [Qwen License](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct/blob/main/LICENSE).

---

<p align="center">
  <b>Desenvolvido por Bruno Simoes | RTX 3070 8GB | 2026</b><br>
  <i>Fine-tuning eficiente para democratizar IA matemática</i>
</p>
