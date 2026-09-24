# AGENTS.md

Este projeto é uma plataforma experimental para compressão de gradientes em PyTorch DDP.

## Invariantes arquiteturais

1. CPU + Gloo deve continuar sendo uma configuração de primeira classe.
2. O loop de treinamento não deve conhecer detalhes de FPGA.
3. Compressão deve ficar isolada do restante do treinamento.
4. Hooks registrados com `DistributedDataParallel.register_comm_hook()` devem:
   - receber `(state, GradBucket)`;
   - devolver `torch.futures.Future[torch.Tensor]`;
   - devolver um tensor compatível com `bucket.buffer()`;
   - preservar a semântica de média dos gradientes quando essa for a política do experimento.
5. Não adicionar barreiras dentro do caminho crítico do hook sem justificativa experimental.
6. Preferir coletivas assíncronas quando isso não prejudicar correção.
7. Não assumir CUDA/NCCL.
8. O projeto deve continuar executando em Python 3 + CPU + Gloo.
9. Mudanças em Top-K devem manter testes de:
   - seleção;
   - residual/error feedback;
   - empacotamento/desempacotamento;
   - preservação exata dos índices INT32.
10. Payload lógico não deve ser apresentado como tráfego físico real.

## Organização

- `train.py`: orchestration e loop de treinamento.
- `distributed.py`: inicialização/finalização do process group.
- `data.py`: dataset e DistributedSampler.
- `models.py`: modelos.
- `hooks/`: políticas de comunicação.
- `metrics.py`: registro de resultados.

Evite concentrar lógica de compressor dentro de `train.py`.

## FPGA

A integração FPGA futura deve entrar atrás de uma interface explícita.

Primeira arquitetura:

```text
GradBucket
  -> host buffer
  -> FPGA compressor
  -> host compressed payload
  -> Gloo
```

Não remova Gloo na primeira integração.

Uma segunda etapa pode avaliar comunicação direta FPGA/rede.

## Desenvolvimento

Antes de concluir alterações:

```bash
python -m compileall src
pytest -q
```

Para mudanças distribuídas, também validar:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m ddp_gradient_compression.train \
  --dataset synthetic \
  --hook dense \
  --train-subset 1024 \
  --test-subset 512 \
  --epochs 1
```

e, quando aplicável:

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

## Métricas

Ao acrescentar otimizações, separar sempre que possível:

- tempo de forward;
- tempo de backward;
- tempo de compressão;
- tempo de comunicação;
- tempo de descompressão;
- tempo total por step;
- payload lógico;
- tráfego real, quando medido externamente;
- accuracy/loss.

Não interpretar redução de bytes como speedup sem medição de tempo.
