# Relatório Técnico — Projeto Integrador CloudTech

Dataset analisado: `latencia_sistemas_cloud.csv` (15.000 registros).

## Fase 1 — Diagnóstico Inicial

### 1) Sumário estatístico da latência por provedor

| Provedor | n | Média (ms) | Mediana (ms) | Desvio Padrão | Q1 | Q3 | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| AWS | 5093 | 33.030 | 15.110 | 34.450 | 9.770 | 61.370 | 104.30% |
| Azure | 4985 | 32.661 | 15.110 | 34.228 | 9.730 | 59.298 | 104.80% |
| GCP | 4922 | 33.150 | 15.230 | 34.314 | 9.973 | 61.770 | 103.51% |

### 2) Visualização (Histograma e Boxplot)

- Histograma: `outputs/histograma_latencia.svg`
- Boxplot: `outputs/boxplot_latencia.svg`
- Outliers (regra de Tukey, global): 9 de 15000 (0.06%).
- Simetria: distribuição claramente assimétrica à direita (cauda longa para altas latências).

## Fase 2 — Modelagem e Riscos

1. **P(throughput < 200 Mbps)** assumindo Normal(551.868, 259.495²):
   - z = (200 - 551.868)/259.495 = -1.356
   - Probabilidade = Φ(z) = **0.0876** (8.76%)
2. **Falha crítica** (Binomial em 1.000 requisições):
   - p histórico de erro = 0.00920 (0.920%)
   - Esperança E[X] = np = **9.200 erros**
   - Variância Var(X) = np(1-p) = **9.115**
3. **Z-score para latência de 150 ms**:
   - Média global = 32.947, DP global = 34.330
   - z = (150 - 32.947)/34.330 = **3.410**
   - Interpretação: valor extremamente acima da média (evento de cauda).

## Fase 3 — Inferência e Decisão

1. **IC 95% para latência média em sa-east-1**:
   - n=3835, média=88.827, DP=16.094
   - IC95% = **[88.318, 89.336] ms**
2. **Teste de hipótese: AWS < GCP (α=1%)**
   - H0: μ_AWS ≥ μ_GCP
   - H1: μ_AWS < μ_GCP
   - Estatística (aprox. Welch): t=-0.175, p-valor unilateral≈0.430407
   - Decisão: **Não rejeitamos H0** (nível de 1%).
3. **Qui-Quadrado de independência: erro x região**
   - χ²=2.329, gl=3, p-valor=0.506921
   - Conclusão: **sem evidência de dependência a 1%**.
   - Região com maior excesso absoluto de erros vs esperado: **ap-northeast-1** (obs=40, esp=34.56).

## Fase 4 — Regressão e Previsão

1. **Regressão linear simples**: latency_ms ~ throughput_mbps
   - latency = 32.5237 + (0.000766) * throughput
   - R² = 0.0000
   - Interpretação: +100 Mbps → variação esperada de **0.077 ms** na latência.
2. **Regressão múltipla** com dummies de provedor (baseline=GCP)
   - latency = 32.7255 + (0.000769)*throughput + (-0.1191)*AWS + (-0.4903)*Azure
   - R² múltiplo = 0.0001 (vs simples 0.0000)
   - Provedor melhora explicação além da velocidade? **Sim**.

## Fase 5 — Storytelling para Diretoria

### 1) O Problema
A operação sul-americana apresentava queixas de instabilidade e possível aumento de latência/erros.
### 2) A Descoberta
Os testes mostraram que há **sem evidência de dependência a 1%**, com destaque para `ap-northeast-1` no excesso de erros. No teste direto AWS vs GCP, não houve evidência estatística suficiente (α=1%) para afirmar que a AWS tem latência média menor que a GCP.
### 3) A Prova (gráfico-chave)
Use o boxplot (`outputs/boxplot_latencia.svg`) para evidenciar a cauda longa e outliers de latência, e a concentração central da maioria das medições.
### 4) A Recomendação
Para expansão em `sa-east-1`, recomenda-se priorizar **GCP**, com latência média estimada de **88.542 ms** e IC95% **[87.650, 89.435] ms**, taxa média de erro **1.021%** (n=1273).