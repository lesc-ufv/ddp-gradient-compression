# DDP Gradient Compression Lab

Projeto experimental em Python 3 para estudar **treinamento distribuído síncrono (S-SGD)** com
**PyTorch DistributedDataParallel (DDP)** e **`register_comm_hook()`**, inicialmente em CPUs
com backend **Gloo**.

O projeto foi estruturado para servir como base de pesquisa e para continuar a evolução dos experimentos. Ele contém:

- baseline DDP padrão (`dense`);
- compressão FP16 usando hook oficial do PyTorch;
- Top-K sparsification com **error feedback**;
- PowerSGD usando o hook oficial do PyTorch;
- métricas de tempo, acurácia e payload lógico;
- execução single-node e multi-node com `torchrun`;
- testes unitários para o compressor Top-K;
- documentação da arquitetura e dos pontos de extensão para FPGA.

> Objetivo principal: separar claramente **treinamento**, **compressão** e **comunicação**, para
> que futuramente o compressor em software possa ser substituído por uma implementação FPGA
> sem reescrever o loop de treinamento.

---

## 1. Arquitetura

Fluxo normal do DDP:

```text
DataLoader
    |
    v
forward
    |
    v
loss
    |
    v
backward
    |
    v
DDP gradient buckets
    |
    v
communication hook
    |
    v
gradient synchronization
    |
    v
optimizer.step()
```

Com o hook Top-K:

```text
                         BACKWARD
                            |
                            v
                     GradBucket pronto
                            |
                            v
                  topk_hook(state, bucket)
                            |
                +-----------+-----------+
                |                       |
                v                       v
          gradient local          residual anterior
                |                       |
                +-----------+-----------+
                            |
                            v
                     error feedback
                            |
                            v
                          Top-K
                            |
                            v
                  values + indices
                            |
                            v
                  Gloo all_gather
                            |
                            v
                      reconstrução
                            |
                            v
                    gradient médio
                            |
                            v
                       Future[Tensor]
                            |
                            v
                           DDP
```

O DDP continua responsável por:

- criar e gerenciar buckets;
- detectar quando um bucket está pronto;
- chamar o hook;
- aguardar o `Future[Tensor]`;
- escrever o resultado da sincronização de volta nos gradientes.

O projeto altera apenas a política de comunicação.

---

## 2. Por que usar `register_comm_hook()`?

No DDP tradicional o PyTorch sincroniza buckets de gradientes automaticamente.

A API:

```python
ddp_model.register_comm_hook(state, hook)
```

substitui a operação padrão para cada `GradBucket`.

A assinatura esperada é:

```python
hook(
    state: object,
    bucket: torch.distributed.GradBucket,
) -> torch.futures.Future[torch.Tensor]
```

Regras importantes:

1. o hook é chamado quando um bucket fica pronto;
2. deve retornar um `Future`;
3. o `Future` deve produzir **um único tensor**;
4. esse tensor deve ser compatível com o buffer original do bucket;
5. quando um hook customizado é registrado, o código do hook deve cuidar da semântica de
   redução/média desejada;
6. o hook deve ser registrado antes do primeiro `backward()`.

Referência:

https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html

---

## 3. Estrutura do projeto

```text
ddp-gradient-compression/
|
+-- README.md
+-- AGENTS.md
+-- requirements.txt
+-- pyproject.toml
+-- .gitignore
|
+-- src/
|   +-- ddp_gradient_compression/
|       +-- __init__.py
|       +-- config.py
|       +-- distributed.py
|       +-- data.py
|       +-- models.py
|       +-- metrics.py
|       +-- train.py
|       |
|       +-- hooks/
|           +-- __init__.py
|           +-- factory.py
|           +-- topk.py
|
+-- scripts/
|   +-- run_local_dense.sh
|   +-- run_local_topk.sh
|   +-- run_multinode_example.sh
|
+-- tests/
    +-- test_topk.py
```

---

## 4. Requisitos

### Software

- Linux recomendado;
- Python >= 3.10;
- PyTorch com suporte a `torch.distributed`;
- torchvision;
- conectividade TCP entre os nós para Gloo.

O projeto foi pensado primeiro para **CPU + Gloo**.

### Hardware inicial

Uma configuração apropriada é:

```text
4 nós
1 rank por nó
CPU em cada nó
Gloo sobre TCP/Ethernet
```

As FPGAs podem ser adicionadas depois no caminho do compressor.

---

## 5. Instalação

Crie um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Instalação genérica:

```bash
pip install -r requirements.txt
pip install -e .
```

Em um cluster CPU-only pode ser preferível instalar explicitamente os wheels CPU do PyTorch
conforme a recomendação oficial da versão escolhida e depois instalar as demais dependências.

Verifique:

```bash
python -c "import torch; print(torch.__version__); print(torch.distributed.is_available())"
```

Também é útil:

```bash
python -m torch.utils.collect_env
```

---

## 6. Dataset

O exemplo usa MNIST.

### NFS não é obrigatório

Cada nó pode ter sua própria cópia local:

```text
node0: /data/mnist
node1: /data/mnist
node2: /data/mnist
node3: /data/mnist
```

O `DistributedSampler` faz a **divisão lógica** das amostras entre os ranks.

O projeto tem a opção:

```bash
--download
```

Quando habilitada, o processo com `LOCAL_RANK=0` em **cada nó** faz o download local antes do
treinamento.

Em cluster sem acesso à Internet, copie previamente o diretório do MNIST para o mesmo caminho
em todos os nós e execute sem `--download`.

---


### Dataset sintético para smoke tests

Para validar DDP sem Internet, NFS ou download de dados:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m ddp_gradient_compression.train \
  --dataset synthetic \
  --hook topk \
  --topk-ratio 0.01 \
  --train-subset 1024 \
  --test-subset 512 \
  --epochs 1
```

O dataset sintético tem formato MNIST (`1x28x28`) e é gerado deterministicamente em cada rank.
Ele serve apenas para **validação funcional/CI**, não para conclusões de convergência ou desempenho
de modelos reais.


## 7. Primeiro teste: 1 processo

Mesmo para um único processo, use `torchrun`:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  -m ddp_gradient_compression.train \
  --hook dense \
  --download
```

Esse teste valida:

- importação do projeto;
- dataset;
- inicialização do process group;
- DDP;
- loop de treinamento;
- geração do CSV.

---

## 8. Single-node com 4 workers

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  -m ddp_gradient_compression.train \
  --hook dense \
  --download
```

Nesse caso:

```text
rank 0
rank 1
rank 2
rank 3

WORLD_SIZE = 4
```

Todos executam na mesma máquina.

---

## 9. Multi-node: 4 nós, 1 rank por nó

Suponha:

```text
node0 = 10.0.0.10
node1 = 10.0.0.11
node2 = 10.0.0.12
node3 = 10.0.0.13
```

Escolha o endpoint de rendezvous:

```text
10.0.0.10:29500
```

Execute **o mesmo comando nos quatro nós**:

```bash
torchrun \
  --nnodes=4 \
  --nproc-per-node=1 \
  --rdzv-id=ddp-exp-001 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=10.0.0.10:29500 \
  -m ddp_gradient_compression.train \
  --hook dense \
  --data-dir /data/mnist
```

O `torchrun` atribui os ranks globais.

Com 1 processo por nó:

```text
node0 -> rank 0
node1 -> rank 1
node2 -> rank 2
node3 -> rank 3
```

`WORLD_SIZE=4`.

O endereço de rendezvous serve para os processos se encontrarem. Ele não transforma o rank 0
em um parameter server.

Referência:

https://docs.pytorch.org/docs/main/elastic/run.html

---

## 10. Selecionando a interface de rede do Gloo

Em máquinas com várias interfaces, defina explicitamente a interface correta.

Exemplo:

```bash
export GLOO_SOCKET_IFNAME=enp1s0
```

Depois execute `torchrun`.

Verifique conectividade entre todos os nós antes do treinamento.

---

## 11. Threads de CPU

Para começar com 1 rank por nó e CPUs multi-core:

```bash
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
```

A quantidade ideal deve ser medida experimentalmente.

Evite começar com muitos ranks por nó e muitos threads por rank simultaneamente, pois isso
pode causar oversubscription.

---

# 12. Hooks disponíveis

Use:

```bash
--hook dense
--hook fp16
--hook topk
--hook powersgd
```

## 12.1 `dense`

Usa o hook oficial de all-reduce do PyTorch.

```text
GradBucket FP32
      |
      v
all_reduce
      |
      v
gradiente médio
```

É o baseline.

Exemplo:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m ddp_gradient_compression.train \
  --hook dense \
  --download
```

---

## 12.2 `fp16`

Usa o hook oficial `fp16_compress_hook`.

Conceitualmente:

```text
FP32 bucket
    |
    v
FP16
    |
    v
all_reduce
    |
    v
FP32
```

Exemplo:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m ddp_gradient_compression.train \
  --hook fp16 \
  --download
```

Esse é um baseline de redução de precisão simples.

---

## 12.3 `topk`

Implementação própria de:

```text
Top-K sparsification
+
error feedback
+
all_gather
```

Exemplo com 1%:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m ddp_gradient_compression.train \
  --hook topk \
  --topk-ratio 0.01 \
  --download
```

### Algoritmo

Para gradiente local `g_t` e residual `e_t`:

```text
u_t = g_t + e_t
```

Selecionamos os K elementos de maior magnitude:

```text
C(u_t) = TopK(u_t)
```

O novo residual é:

```text
e_(t+1) = u_t - C(u_t)
```

Os elementos transmitidos são:

```text
(value, index)
```

### Payload

Para FP32 + INT32:

```text
value = 4 bytes
index = 4 bytes
```

Se a razão é `r`, o payload lógico aproximado por bucket é:

```text
8 * r * N bytes
```

O bucket denso FP32 teria:

```text
4 * N bytes
```

Logo, ignorando metadados e protocolo:

```text
compression_ratio ~= 1 / (2r)
```

Para `r=0.01`:

```text
~50x
```

### Representação usada no código

O projeto empacota:

```text
[FP32 values][INT32 indices reinterpretados como bits FP32]
```

em um único tensor FP32.

Isto permite usar **um único `all_gather` assíncrono por bucket**.

Os índices não são convertidos numericamente para float. Seus bits INT32 são reinterpretados
como FP32 durante o transporte e reinterpretados de volta após a comunicação.

### Por que `all_gather`?

Cada worker pode selecionar índices diferentes.

Exemplo:

```text
rank0 -> indices [1, 7, 20]
rank1 -> indices [2, 7, 40]
```

Não é possível simplesmente aplicar um all-reduce denso sobre esses pares sem uma estratégia
adicional.

A implementação coleta os pares esparsos de todos os workers e reconstrói a soma localmente.

### Limitação importante

**Menos bytes não implica automaticamente menor tempo de treinamento.**

Top-K adiciona:

- `abs`;
- seleção Top-K;
- geração de índices;
- armazenamento de residual;
- empacotamento;
- `all_gather`;
- reconstrução esparsa.

Por isso devem ser medidos:

```text
tempo de step
tempo de época
payload
acurácia
convergência
```

e, futuramente:

```text
tempo de compressão
tempo de DMA
tempo de comunicação
```

---

## 12.4 `powersgd`

Usa a implementação oficial do PyTorch de PowerSGD.

Exemplo:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m ddp_gradient_compression.train \
  --hook powersgd \
  --powersgd-rank 1 \
  --powersgd-start-iter 10 \
  --download
```

O PowerSGD é um baseline particularmente importante para comparar com uma futura
implementação em FPGA, pois seu núcleo é dominado por operações matriciais.

Documentação:

https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html

---

# 13. Buckets do DDP

O tamanho do bucket pode ser configurado:

```bash
--bucket-cap-mb 1
```

O valor pequeno é propositalmente útil para experimentação com modelos pequenos, pois aumenta
a chance de existirem vários buckets e deixa visível a granularidade do hook.

Em modelos reais, o valor ótimo deve ser medido.

Fluxo:

```text
backward
   |
   +--> bucket 0 pronto --> hook --> comunicação
   |
   +--> bucket 1 pronto --> hook --> comunicação
   |
   +--> bucket 2 pronto --> hook --> comunicação
```

Isso permite sobreposição entre backward e comunicação.

---

# 14. Modelo e batch

O exemplo usa um MLP para MNIST.

Por padrão:

```text
784 -> 512 -> 512 -> 10
```

O batch passado por CLI é **batch local por rank**.

Por exemplo:

```bash
--batch-size 64
```

com:

```text
WORLD_SIZE=4
```

produz aproximadamente:

```text
global batch = 64 * 4 = 256
```

Essa diferença precisa ser considerada ao comparar execuções com números diferentes de
workers.

---

# 15. Subsets para testes rápidos

Por padrão o projeto pode trabalhar com subconjuntos:

```bash
--train-subset 12000
--test-subset 2000
```

Para usar o dataset inteiro:

```bash
--train-subset 0
--test-subset 0
```

---

# 16. Resultados

O rank 0 grava CSV em:

```text
results/
```

Exemplo de colunas:

```text
epoch
hook
world_size
train_loss
test_loss
test_accuracy
epoch_seconds
dense_reference_mb_per_worker
logical_payload_mb_per_worker
logical_compression_ratio
```

Para `Top-K`, o payload é contado a partir dos buckets efetivamente processados pelo hook.

Para `dense` e `fp16`, o projeto calcula uma referência lógica a partir do número de parâmetros
e do número de passos.

Para PowerSGD, o projeto não tenta inferir os bytes reais do algoritmo interno; o campo pode
ficar vazio. Para estudo rigoroso de PowerSGD, instrumente explicitamente as operações do hook
ou use profiling de rede.

---

# 17. Experimento inicial recomendado

Execute:

```text
A. dense
B. fp16
C. topk 10%
D. topk 1%
E. topk 0.1%
F. powersgd rank 1
```

Mantendo constantes:

```text
modelo
dataset
seed
batch local
número de workers
threads de CPU
rede
```

Colete:

```text
accuracy
loss
epoch time
samples/s
logical bytes
compression ratio
```

Depois repita múltiplas vezes para separar ruído de sistema de diferenças reais.

---

# 18. Caminho para FPGA

A primeira integração deve preservar o restante do pipeline:

```text
DDP
 |
 v
GradBucket
 |
 v
comm hook
 |
 v
CPU compressor
```

vira:

```text
DDP
 |
 v
GradBucket
 |
 v
comm hook
 |
 v
host -> FPGA DMA
 |
 v
FPGA compressor
 |
 v
FPGA -> host
 |
 v
Gloo
```

O primeiro objetivo não deve ser remover Gloo. O objetivo é comparar:

```text
CPU compression
vs
FPGA compression
```

mantendo constante:

```text
DDP
dataset
modelo
workers
rede
coletiva
```

Depois é possível estudar uma segunda arquitetura:

```text
CPU
 |
 v
FPGA
 |
 v
compressão
 |
 v
rede diretamente
```

reduzindo o retorno ao host.

---

# 19. Interface sugerida para o futuro FPGA

Uma direção de evolução é criar:

```python
class Compressor:
    def compress(self, bucket):
        ...

    def decompress(self, payload):
        ...
```

e implementações:

```text
TopKCPUCompressor
QuantizationCPUCompressor
PowerSGDCompressor
FPGACompressor
```

O hook passa a ser um adaptador entre:

```text
DDP GradBucket
    |
    v
Compressor
    |
    v
Communication backend
```

Isso mantém o projeto modular.

---

# 20. Duas FPGAs por nó

Não comece com 2 ranks por nó apenas porque há duas FPGAs.

Primeira configuração recomendada:

```text
1 rank por nó
2 FPGAs como aceleradores auxiliares
```

Possíveis estratégias futuras:

```text
FPGA0 -> bucket N
FPGA1 -> bucket N+1
```

ou:

```text
FPGA0 -> primeira metade do bucket
FPGA1 -> segunda metade
```

Mais tarde pode ser avaliado:

```text
2 ranks por nó
1 FPGA por rank
```

mas isso muda simultaneamente:

- world size;
- batch global;
- disputa por CPU;
- disputa por memória;
- comportamento de comunicação.

Por isso não é a primeira configuração recomendada.

---

# 21. Profiling futuro

Pontos importantes para instrumentação:

```text
T_forward
T_backward
T_compress
T_dma_host_to_fpga
T_fpga
T_dma_fpga_to_host
T_collective
T_step
```

A condição básica para a compressão ser útil é:

```text
T_compress
+ T_communication_compressed
+ T_decompress

<

T_communication_dense
```

Com FPGA:

```text
T_H2D
+ T_FPGA
+ T_D2H
+ T_communication_compressed
+ T_decompress

<

T_communication_dense
```

---

# 22. Testes

Execute:

```bash
pytest -q
```

Os testes atuais cobrem principalmente a parte matemática/serialização do Top-K sem exigir um
cluster distribuído.

Antes de adicionar um backend FPGA, mantenha esses testes e acrescente:

- round-trip CPU -> FPGA -> CPU;
- equivalência de índices/valores;
- tratamento de buckets pequenos;
- razão de compressão;
- error feedback;
- determinismo quando aplicável.

---

# 23. Desenvolvimento com Codex

Leia também:

```text
AGENTS.md
```

O arquivo contém invariantes arquiteturais que devem ser preservados ao modificar o projeto.

Tarefas naturais para as próximas etapas:

1. adicionar profiling por bucket;
2. separar `Compressor` de `CommunicationStrategy`;
3. implementar quantização INT8;
4. implementar Top-K assíncrono com melhor overlap;
5. adicionar benchmark de throughput;
6. integrar PowerSGD aos relatórios;
7. criar interface de runtime FPGA;
8. adicionar mock FPGA para desenvolvimento sem hardware;
9. medir PCIe/DMA;
10. implementar execução via Slurm;
11. coletar múltiplas repetições automaticamente;
12. produzir gráficos de convergência vs bytes transmitidos.

---

# 24. Cuidados experimentais

## Não confundir payload lógico com tráfego real

O campo `logical_payload_mb_per_worker` mede os bytes representados pelos tensores do
algoritmo.

Ele não inclui necessariamente:

- cabeçalhos TCP/IP;
- overhead do Gloo;
- algoritmos internos da coletiva;
- retransmissões;
- buffers temporários;
- tráfego de rendezvous.

Para medir rede real use também ferramentas externas ou contadores da interface.

## Top-K não necessariamente usa a melhor coletiva

A implementação atual usa `all_gather` por simplicidade e correção.

Um algoritmo mais avançado pode usar:

- reduce-scatter;
- all-to-all;
- sketches;
- agregação hierárquica;
- comunicação customizada.

## Modelo pequeno não representa escala real

MNIST é para validação funcional.

Depois use um workload maior, por exemplo:

```text
CIFAR-10 + CNN
CIFAR-10 + ResNet-18
modelo sintético com gradientes grandes
gradient replay
```

O último é especialmente útil para avaliar o subsistema de comunicação sem que o custo do
forward/backward domine o experimento.

---

# 25. Referências técnicas

PyTorch DistributedDataParallel:

https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html

PyTorch DDP communication hooks:

https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html

torchrun:

https://docs.pytorch.org/docs/main/elastic/run.html

PowerSGD:

Vogels et al.  
PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization  
NeurIPS 2019.

Error feedback:

Karimireddy et al.  
Error Feedback Fixes SignSGD and other Gradient Compression Schemes  
ICML 2019.

Deep Gradient Compression:

Lin et al.  
Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training  
ICLR 2018.

---

# 26. Escopo atual

O projeto atual é uma **plataforma experimental**, não uma implementação de produção.

Ele foi deliberadamente construído para responder perguntas como:

```text
Quanto Top-K reduz o payload?

Qual o efeito na convergência?

Qual o overhead do compressor?

Quando compressão deixa de compensar?

Quanto uma FPGA reduz o tempo do compressor?

Qual o impacto do tamanho do bucket?

Quanto conseguimos sobrepor backward, compressão e comunicação?
```

Essas perguntas devem orientar as próximas evoluções do código.
