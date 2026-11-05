# Backend ACCL (experimental)

`--backend accl` registra um `ProcessGroup` próprio do PyTorch. As coletivas
suportadas usam PyACCL; não existe fallback automático para Gloo, MPI ou Store
para transportar gradientes. Gloo permanece como backend padrão independente.
O modelo, os compressores e o otimizador continuam na CPU.

**Estado de validação:** testes do adaptador PyACCL com um driver controlado e
testes do ProcessGroup/DDP com dois processos e transporte de teste. Estes testes
não executam o emulador oficial nem validam hardware U55C. A execução real depende
do runtime e de um design ACCL compatível. Não interpretar os tempos dos testes
como desempenho ACCL.

## Operações e contrato

- `broadcast`, `all_reduce(SUM)`, `all_gather` de tensores de mesmo tamanho,
  `all_gather_into_tensor` e `barrier`.
- Listas de tensores e all-reduce coalescido são despachados em ordem.
- Buffers CPU contíguos FP16/FP32/FP64/INT32/INT64. O bitstream também deve ter
  os plugins aritméticos correspondentes. Outros tipos/reduções são rejeitados.
- `Work.wait()` e `Work.get_future()`: resultado visível na CPU após sincronização
  device-to-host. A média continua sendo responsabilidade dos hooks.
- Somente o comunicador WORLD, uma instância do backend por processo e uma placa
  dedicada por rank. Subgrupos e compartilhamento de uma placa entre ranks não
  são suportados nesta primeira versão.
- Uma thread de I/O possui o driver, inclusive seu socket de emulação. Ela executa
  as coletivas em ordem, sem inserir barreiras no caminho dos hooks. Chamadas
  `async_op=True` retornam ao chamador antes da conclusão.
- Timeouts e falhas tornam o grupo inutilizável e completam os futures com erro.
  Um timeout no host não cancela uma chamada XRT/PyACCL pendente. Os buffers não
  são liberados enquanto essa chamada estiver ativa; reinicie o job e verifique
  o estado da placa antes de reutilizá-la.

## Driver e compatibilidade

A API foi conferida no código de [Xilinx/pyaccl](https://github.com/Xilinx/pyaccl),
commit `ff876197ac476f302ed485269b6926ae11df4b81` (API 0.2), em particular
`accl()`, `allocate`, `allreduce`, `allgather`, `bcast`, `check_return_value` e
os métodos de sincronização dos buffers. O contrato PyTorch foi testado localmente
com PyTorch 2.14.1; compatibilidade de outras versões precisa ser validada.

Instale o runtime no ambiente apropriado do cluster, com PYNQ/XRT compatíveis:

```bash
python -m pip install 'pyaccl @ git+https://github.com/Xilinx/pyaccl.git@ff876197ac476f302ed485269b6926ae11df4b81'
```

Não se trata de instalar apenas uma biblioteca Python: hardware requer o design
CCLO, host controller, plugins aritméticos, stack TCP/UDP, CMAC e configuração de
rede compatíveis. O caminho `xclbin` deve apontar para um design ACCL, não para
um bitstream arbitrário da U55C. Consulte o [guia do ACCL](https://github.com/Xilinx/ACCL/blob/main/INSTALL.md).
A revisão do emulador/bitstream deve corresponder à revisão do driver; a versão
mais recente do ACCL não é presumida compatível com qualquer PyACCL antigo.

## Emulador oficial

Prepare e inicie o emulador ACCL segundo a documentação do fornecedor. Com
`cclo_emu` disponível, PyACCL documenta:

```bash
pyaccl-emulate -n 2
```

Em outro terminal, na raiz deste projeto:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export ACCL_CONFIG="$PWD/configs/accl-emulator-2ranks.json"
export OMP_NUM_THREADS=2

torchrun --standalone --nproc-per-node=2 \
  -m ddp_gradient_compression.train --backend accl \
  --dataset synthetic --hook dense \
  --train-subset 1024 --test-subset 512 --epochs 1
```

O JSON lista explicitamente os sockets ZMQ dos dois ranks. Ajuste-os à sessão do
emulador. Recrie/reset a sessão entre jobs conforme os requisitos do driver: o
PyACCL rejeita um CCLO já configurado. Depois valide `fp16`, `topk` e `powersgd`;
para este último use `--powersgd-start-iter 2` para exercitar a compressão em
uma época curta. Não há modo `mock` selecionável pelo treinamento.

## Cluster: quatro nós, duas U55C por nó

Comece com quatro ranks, uma placa por nó. A segunda placa permanece disponível
para uma etapa posterior. Copie e edite `configs/accl-u55c-4nodes.example.json`:

- `ranks`: descritores ACCL em ordem global, com IP da FPGA, porta, session ID e
  tamanho máximo de segmento. Os IPs `192.0.2.x` do exemplo são placeholders.
- `devices`: hostname, índice local da placa, índice CCLO e caminho do xclbin.
  O hostname deve corresponder a `socket.gethostname()` ou `socket.getfqdn()`.
- `protocol`, `nbufs`, `bufsize`: devem corresponder ao design e aos buffers da
  aplicação. A versão inicial usa chunks conservadores de até um segmento.
- `driver_factory`: função Python do cluster, no formato `modulo:funcao`, que
  inicializa o driver, a placa e a rede.

### Adaptador do cluster: requisito para hardware

O módulo `cluster_accl` **não é fornecido como uma implementação pronta**:
a configuração da stack Ethernet depende do bitstream e das versões instaladas,
que ainda não foram informados. O backend falha explicitamente se ele estiver
ausente. Um caminho de bitstream e uma tabela de IPs, isoladamente, não programam
a stack Ethernet.

Implemente a função com este contrato:

```python
def create_driver(config, global_rank, world_size):
    # Carregar o xclbin da placa exclusiva deste rank.
    # Configurar IP/MAC, portas, ARP/conexões e verificar o link da stack do design.
    # Criar/configurar um driver compatível com a API pyaccl.accl 0.2.
    # Aplicar o timeout nativo nas unidades da revisão do CCLO.
    # Retornar o driver pronto para executar coletivas.
    ...
```

A função recebe `ACCLConfig`, e `config.devices[global_rank]` identifica a placa.
O objeto retornado deve oferecer `allocate`, `allreduce`, `allgather`, `bcast`,
`barrier`, `check_return_value` e buffers com `sync_to_device/sync_from_device`.
As três coletivas de dados devem aceitar `run_async=True` e retornar um handle
com `wait()`. O método opcional `close()` permite liberar recursos do adaptador.
Não deve retornar um transporte Gloo disfarçado de ACCL.

O construtor PyACCL pode abrir conexões TCP durante sua inicialização. A stack
Ethernet precisa estar configurada antes dessa etapa; não basta alterar os IPs
após o construtor retornar. O adaptador do cluster é o local dessa integração,
sem alterações no loop de treinamento.

Uma vez que o adaptador esteja instalado e os quatro hosts tenham a mesma
configuração global, execute em cada nó (mudando `NODE_RANK` de 0 a 3):

```bash
export ACCL_CONFIG=/caminho/accl-cluster.json
export MASTER_ADDR=IP_DO_HOST_0
export NODE_RANK=0
bash scripts/run_multinode_accl.sh
```

O endereço de rendezvous é o IP da CPU, não o IP da FPGA. O launcher usa
rendezvous TCP estático para garantir que a ordem dos ranks corresponda ao JSON.
`--backend accl` transporta também a sincronização inicial do modelo, métricas e
barreiras via ACCL. O TCPStore do PyTorch continua usado para rendezvous/controle.

Para outros hooks:

```bash
HOOK=topk bash scripts/run_multinode_accl.sh --topk-ratio 0.01
HOOK=powersgd bash scripts/run_multinode_accl.sh --powersgd-start-iter 2
```

Usar ambas as placas é possível com oito ranks e uma placa dedicada por processo,
mas requer um JSON com oito entradas e `NPROC_PER_NODE=2`; isso muda o world size e
o batch global. Um rank distribuindo trabalho por duas placas ainda não está
implementado. Esta configuração de oito ranks também precisa ser validada no cluster.

## Payload e medições

O backend não aplica conversão FP32→FP16 implícita. Isso preserva os índices INT32
reinterpretados como FP32 pelo Top-K, inclusive padrões de NaN. O hook FP16 pode
continuar convertendo explicitamente seus gradientes antes da coletiva.

Os buffers são reutilizados por dtype e as transferências são divididas em chunks
com padding de 64 bytes. Os contadores dos hooks continuam medindo payload lógico
da representação original; não incluem padding, protocolos, DMA ou número de
saltos do algoritmo coletivo. Não constituem medição de tráfego físico.

CSVs e nomes dos arquivos incluem `backend` e `backend_mode` (hardware/emulator).
Os gráficos selecionam as execuções mais recentes separadamente por backend/modo,
para não substituir uma referência Gloo por uma execução emulada do ACCL.

Antes de publicar resultados: valide os quatro hooks em dois e quatro participantes,
compare gradientes/loss com Gloo, teste INT32 e mudanças de buckets, e meça DMA,
comunicação e tempo total separadamente. A fila serial e os chunks conservadores
priorizam correção; esta implementação não promete saturar o switch 100G.
