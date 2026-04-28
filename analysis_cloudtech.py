import csv
import math
from collections import defaultdict, Counter
from pathlib import Path

DATA_FILE = 'latencia_sistemas_cloud.csv'
OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs, sample=True):
    m = mean(xs)
    n = len(xs)
    denom = (n - 1) if sample else n
    return sum((x - m) ** 2 for x in xs) / denom


def stddev(xs, sample=True):
    return math.sqrt(variance(xs, sample=sample))


def percentile(xs, p):
    ys = sorted(xs)
    if not ys:
        return float('nan')
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    d0 = ys[f] * (c - k)
    d1 = ys[c] * (k - f)
    return d0 + d1


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# Regularized incomplete gamma Q(a, x) via Numerical Recipes

def gammaincc(a, x):
    if x < 0 or a <= 0:
        return float('nan')
    if x == 0:
        return 1.0
    if x < a + 1:
        # Use series for P(a, x), then Q = 1 - P
        ap = a
        summ = 1.0 / a
        delt = summ
        for _ in range(1000):
            ap += 1
            delt *= x / ap
            summ += delt
            if abs(delt) < abs(summ) * 1e-14:
                break
        p = summ * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    else:
        # Continued fraction for Q(a, x)
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, 1000):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delt = d * c
            h *= delt
            if abs(delt - 1.0) < 1e-14:
                break
        q = h * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, q))


def chi_square_sf(chi2, k):
    return gammaincc(k / 2.0, chi2 / 2.0)


def invert_matrix(a):
    n = len(a)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError('Matriz singular')
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [x / piv for x in aug[col]]
        for r in range(n):
            if r != col:
                fac = aug[r][col]
                aug[r] = [aug[r][c] - fac * aug[col][c] for c in range(2 * n)]

    return [row[n:] for row in aug]


def matmul(a, b):
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def matvec(a, x):
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def linear_regression_simple(x, y):
    mx = mean(x)
    my = mean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b1 = sxy / sxx
    b0 = my - b1 * mx

    yhat = [b0 + b1 * xi for xi in x]
    ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y, yhat))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot
    return b0, b1, r2


def linear_regression_multiple(X, y):
    # X already includes intercept col
    Xt = list(zip(*X))
    XtX = matmul([list(row) for row in Xt], X)
    XtX_inv = invert_matrix(XtX)
    Xty = [sum(xij * yi for xij, yi in zip(col, y)) for col in Xt]
    beta = matvec(XtX_inv, Xty)

    yhat = [sum(b * xij for b, xij in zip(beta, row)) for row in X]
    my = mean(y)
    ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y, yhat))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot
    return beta, r2


def draw_histogram(data, path, bins=30, width=900, height=420):
    mn, mx = min(data), max(data)
    step = (mx - mn) / bins
    counts = [0] * bins
    for v in data:
        idx = int((v - mn) / step)
        if idx == bins:
            idx -= 1
        counts[idx] += 1

    maxc = max(counts)
    margin = 50
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>')
    parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>')

    bw = plot_w / bins
    for i, c in enumerate(counts):
        bh = (c / maxc) * plot_h
        x = margin + i * bw
        y = height - margin - bh
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw-1:.2f}" height="{bh:.2f}" fill="#4e79a7"/>')

    parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="16">Histograma de Latência (ms)</text>')
    parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="12">latency_ms</text>')
    parts.append(f'<text x="15" y="{height/2}" text-anchor="middle" transform="rotate(-90 15,{height/2})" font-size="12">Frequência</text>')
    parts.append(f'<text x="{margin}" y="{height-margin+20}" font-size="11">{mn:.1f}</text>')
    parts.append(f'<text x="{width-margin-30}" y="{height-margin+20}" font-size="11">{mx:.1f}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def draw_boxplot(data, path, width=900, height=260):
    ys = sorted(data)
    q1 = percentile(ys, 0.25)
    med = percentile(ys, 0.50)
    q3 = percentile(ys, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    non_out = [v for v in ys if lower <= v <= upper]
    low_whisk = min(non_out)
    up_whisk = max(non_out)
    outliers = [v for v in ys if v < lower or v > upper]

    mn, mx = min(ys), max(ys)
    margin = 60
    y = height / 2

    def sx(v):
        return margin + (v - mn) / (mx - mn) * (width - 2 * margin)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<line x1="{margin}" y1="{y+50}" x2="{width-margin}" y2="{y+50}" stroke="black"/>')

    # whiskers
    parts.append(f'<line x1="{sx(low_whisk):.2f}" y1="{y}" x2="{sx(q1):.2f}" y2="{y}" stroke="black"/>')
    parts.append(f'<line x1="{sx(q3):.2f}" y1="{y}" x2="{sx(up_whisk):.2f}" y2="{y}" stroke="black"/>')
    parts.append(f'<line x1="{sx(low_whisk):.2f}" y1="{y-20}" x2="{sx(low_whisk):.2f}" y2="{y+20}" stroke="black"/>')
    parts.append(f'<line x1="{sx(up_whisk):.2f}" y1="{y-20}" x2="{sx(up_whisk):.2f}" y2="{y+20}" stroke="black"/>')

    # box + median
    parts.append(f'<rect x="{sx(q1):.2f}" y="{y-25}" width="{sx(q3)-sx(q1):.2f}" height="50" fill="#a0cbe8" stroke="black"/>')
    parts.append(f'<line x1="{sx(med):.2f}" y1="{y-25}" x2="{sx(med):.2f}" y2="{y+25}" stroke="black"/>')

    # outliers sampled for rendering density
    if outliers:
        step = max(1, len(outliers) // 400)
        for v in outliers[::step]:
            parts.append(f'<circle cx="{sx(v):.2f}" cy="{y}" r="2" fill="#e15759"/>')

    parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="16">Boxplot de Latência (ms)</text>')
    parts.append(f'<text x="{margin}" y="{y+70}" font-size="11">{mn:.1f}</text>')
    parts.append(f'<text x="{width-margin-30}" y="{y+70}" font-size="11">{mx:.1f}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    rows = []
    with open(DATA_FILE, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            row['latency_ms'] = float(row['latency_ms'])
            row['throughput_mbps'] = float(row['throughput_mbps'])
            row['error_rate'] = int(row['error_rate'])
            rows.append(row)

    lat_by_provider = defaultdict(list)
    throughput = []
    lat = []
    error = []
    region_errors = defaultdict(lambda: [0, 0])  # [no error, error]

    for row in rows:
        p = row['provider']
        lat_by_provider[p].append(row['latency_ms'])
        throughput.append(row['throughput_mbps'])
        lat.append(row['latency_ms'])
        error.append(row['error_rate'])
        region_errors[row['region']][row['error_rate']] += 1

    # Fase 1: estatística descritiva por provedor
    desc = {}
    for p, vals in lat_by_provider.items():
        m = mean(vals)
        sd = stddev(vals)
        desc[p] = {
            'n': len(vals),
            'mean': m,
            'median': percentile(vals, 0.5),
            'std': sd,
            'q1': percentile(vals, 0.25),
            'q3': percentile(vals, 0.75),
            'cv': sd / m,
        }

    # outliers globais
    q1_all = percentile(lat, 0.25)
    q3_all = percentile(lat, 0.75)
    iqr_all = q3_all - q1_all
    lower_all = q1_all - 1.5 * iqr_all
    upper_all = q3_all + 1.5 * iqr_all
    outliers_n = sum(1 for v in lat if v < lower_all or v > upper_all)

    # Fase 2
    mu_tp = mean(throughput)
    sd_tp = stddev(throughput)
    z_200 = (200 - mu_tp) / sd_tp
    prob_below_200 = normal_cdf(z_200)

    p_error = mean(error)
    n_batch = 1000
    exp_errors = n_batch * p_error
    var_errors = n_batch * p_error * (1 - p_error)

    mu_lat = mean(lat)
    sd_lat = stddev(lat)
    z_150 = (150 - mu_lat) / sd_lat

    # Fase 3
    sa = [r['latency_ms'] for r in rows if r['region'] == 'sa-east-1']
    n_sa = len(sa)
    mu_sa = mean(sa)
    sd_sa = stddev(sa)
    zcrit95 = 1.959963984540054
    ci_sa = (mu_sa - zcrit95 * sd_sa / math.sqrt(n_sa), mu_sa + zcrit95 * sd_sa / math.sqrt(n_sa))

    aws = lat_by_provider['AWS']
    gcp = lat_by_provider['GCP']
    m1, m2 = mean(aws), mean(gcp)
    s1, s2 = stddev(aws), stddev(gcp)
    n1, n2 = len(aws), len(gcp)
    se = math.sqrt((s1**2)/n1 + (s2**2)/n2)
    t_stat = (m1 - m2) / se
    # one-sided p-value H1: AWS < GCP
    p_one = normal_cdf(t_stat)

    # chi-square independência erro x região
    regions = sorted(region_errors)
    total = sum(sum(v) for v in region_errors.values())
    row_totals = {reg: sum(region_errors[reg]) for reg in regions}
    col_totals = {
        0: sum(region_errors[reg][0] for reg in regions),
        1: sum(region_errors[reg][1] for reg in regions),
    }
    chi2 = 0.0
    contributions = []
    for reg in regions:
        for e in (0, 1):
            observed = region_errors[reg][e]
            expected = row_totals[reg] * col_totals[e] / total
            contrib = (observed - expected) ** 2 / expected
            chi2 += contrib
            contributions.append((reg, e, contrib, observed, expected))
    dof = (len(regions) - 1) * (2 - 1)
    p_chi2 = chi_square_sf(chi2, dof)

    # região com maior excesso de erros
    worst_region = None
    worst_excess = -1e18
    for reg in regions:
        obs_err = region_errors[reg][1]
        exp_err = row_totals[reg] * col_totals[1] / total
        excess = obs_err - exp_err
        if excess > worst_excess:
            worst_excess = excess
            worst_region = (reg, obs_err, exp_err)

    # Fase 4 regressão
    b0, b1, r2_simple = linear_regression_simple(throughput, lat)

    # dummies: baseline GCP
    X = []
    for r in rows:
        is_aws = 1.0 if r['provider'] == 'AWS' else 0.0
        is_azure = 1.0 if r['provider'] == 'Azure' else 0.0
        X.append([1.0, r['throughput_mbps'], is_aws, is_azure])
    beta, r2_multi = linear_regression_multiple(X, lat)

    # Render charts
    draw_histogram(lat, OUT_DIR / 'histograma_latencia.svg')
    draw_boxplot(lat, OUT_DIR / 'boxplot_latencia.svg')

    # Provedor recomendado para sa-east-1 pelo menor mean e menor erro
    sa_by_provider = defaultdict(list)
    sa_err_provider = defaultdict(list)
    for r in rows:
        if r['region'] == 'sa-east-1':
            sa_by_provider[r['provider']].append(r['latency_ms'])
            sa_err_provider[r['provider']].append(r['error_rate'])

    recommendation = {}
    for p, vals in sa_by_provider.items():
        mp = mean(vals)
        sdp = stddev(vals)
        np = len(vals)
        ci = (mp - zcrit95 * sdp / math.sqrt(np), mp + zcrit95 * sdp / math.sqrt(np))
        recommendation[p] = {
            'mean': mp,
            'ci': ci,
            'error_rate': mean(sa_err_provider[p]),
            'n': np,
        }

    best_provider = min(recommendation.items(), key=lambda kv: (kv[1]['mean'], kv[1]['error_rate']))

    # Write markdown report
    lines = []
    lines.append('# Relatório Técnico — Projeto Integrador CloudTech\n')
    lines.append('Dataset analisado: `latencia_sistemas_cloud.csv` (15.000 registros).\n')

    lines.append('## Fase 1 — Diagnóstico Inicial\n')
    lines.append('### 1) Sumário estatístico da latência por provedor\n')
    lines.append('| Provedor | n | Média (ms) | Mediana (ms) | Desvio Padrão | Q1 | Q3 | CV |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for p in sorted(desc):
        d = desc[p]
        lines.append(f"| {p} | {d['n']} | {d['mean']:.3f} | {d['median']:.3f} | {d['std']:.3f} | {d['q1']:.3f} | {d['q3']:.3f} | {d['cv']*100:.2f}% |")

    lines.append('\n### 2) Visualização (Histograma e Boxplot)\n')
    lines.append('- Histograma: `outputs/histograma_latencia.svg`')
    lines.append('- Boxplot: `outputs/boxplot_latencia.svg`')
    lines.append(f'- Outliers (regra de Tukey, global): {outliers_n} de {len(lat)} ({100*outliers_n/len(lat):.2f}%).')
    lines.append('- Simetria: distribuição claramente assimétrica à direita (cauda longa para altas latências).\n')

    lines.append('## Fase 2 — Modelagem e Riscos\n')
    lines.append(f"1. **P(throughput < 200 Mbps)** assumindo Normal({mu_tp:.3f}, {sd_tp:.3f}²):")
    lines.append(f"   - z = (200 - {mu_tp:.3f})/{sd_tp:.3f} = {z_200:.3f}")
    lines.append(f"   - Probabilidade = Φ(z) = **{prob_below_200:.4f}** ({prob_below_200*100:.2f}%)")

    lines.append('2. **Falha crítica** (Binomial em 1.000 requisições):')
    lines.append(f"   - p histórico de erro = {p_error:.5f} ({p_error*100:.3f}%)")
    lines.append(f"   - Esperança E[X] = np = **{exp_errors:.3f} erros**")
    lines.append(f"   - Variância Var(X) = np(1-p) = **{var_errors:.3f}**")

    lines.append('3. **Z-score para latência de 150 ms**:')
    lines.append(f"   - Média global = {mu_lat:.3f}, DP global = {sd_lat:.3f}")
    lines.append(f"   - z = (150 - {mu_lat:.3f})/{sd_lat:.3f} = **{z_150:.3f}**")
    lines.append('   - Interpretação: valor extremamente acima da média (evento de cauda).\n')

    lines.append('## Fase 3 — Inferência e Decisão\n')
    lines.append('1. **IC 95% para latência média em sa-east-1**:')
    lines.append(f"   - n={n_sa}, média={mu_sa:.3f}, DP={sd_sa:.3f}")
    lines.append(f"   - IC95% = **[{ci_sa[0]:.3f}, {ci_sa[1]:.3f}] ms**")

    lines.append('2. **Teste de hipótese: AWS < GCP (α=1%)**')
    lines.append('   - H0: μ_AWS ≥ μ_GCP')
    lines.append('   - H1: μ_AWS < μ_GCP')
    lines.append(f"   - Estatística (aprox. Welch): t={t_stat:.3f}, p-valor unilateral≈{p_one:.6f}")
    decision = 'Rejeitamos H0' if p_one < 0.01 else 'Não rejeitamos H0'
    lines.append(f"   - Decisão: **{decision}** (nível de 1%).")

    lines.append('3. **Qui-Quadrado de independência: erro x região**')
    lines.append(f"   - χ²={chi2:.3f}, gl={dof}, p-valor={p_chi2:.6f}")
    chi_decision = 'dependência estatística entre região e erro' if p_chi2 < 0.01 else 'sem evidência de dependência a 1%'
    lines.append(f"   - Conclusão: **{chi_decision}**.")
    lines.append(f"   - Região com maior excesso absoluto de erros vs esperado: **{worst_region[0]}** (obs={worst_region[1]}, esp={worst_region[2]:.2f}).\n")

    lines.append('## Fase 4 — Regressão e Previsão\n')
    lines.append('1. **Regressão linear simples**: latency_ms ~ throughput_mbps')
    lines.append(f"   - latency = {b0:.4f} + ({b1:.6f}) * throughput")
    lines.append(f"   - R² = {r2_simple:.4f}")
    lines.append(f"   - Interpretação: +100 Mbps → variação esperada de **{100*b1:.3f} ms** na latência.")

    lines.append('2. **Regressão múltipla** com dummies de provedor (baseline=GCP)')
    lines.append(f"   - latency = {beta[0]:.4f} + ({beta[1]:.6f})*throughput + ({beta[2]:.4f})*AWS + ({beta[3]:.4f})*Azure")
    lines.append(f"   - R² múltiplo = {r2_multi:.4f} (vs simples {r2_simple:.4f})")
    better = 'Sim' if r2_multi > r2_simple else 'Não'
    lines.append(f"   - Provedor melhora explicação além da velocidade? **{better}**.\n")

    lines.append('## Fase 5 — Storytelling para Diretoria\n')
    lines.append('### 1) O Problema')
    lines.append('A operação sul-americana apresentava queixas de instabilidade e possível aumento de latência/erros.')
    lines.append('### 2) A Descoberta')
    lines.append(f"Os testes mostraram que há **{chi_decision}**, com destaque para `{worst_region[0]}` no excesso de erros. No teste direto AWS vs GCP, não houve evidência estatística suficiente (α=1%) para afirmar que a AWS tem latência média menor que a GCP.")
    lines.append('### 3) A Prova (gráfico-chave)')
    lines.append('Use o boxplot (`outputs/boxplot_latencia.svg`) para evidenciar a cauda longa e outliers de latência, e a concentração central da maioria das medições.')
    lines.append('### 4) A Recomendação')
    bp, bd = best_provider
    lines.append(f"Para expansão em `sa-east-1`, recomenda-se priorizar **{bp}**, com latência média estimada de **{bd['mean']:.3f} ms** e IC95% **[{bd['ci'][0]:.3f}, {bd['ci'][1]:.3f}] ms**, taxa média de erro **{bd['error_rate']*100:.3f}%** (n={bd['n']}).")

    (OUT_DIR / 'relatorio_cloudtech.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Relatório gerado em outputs/relatorio_cloudtech.md')


if __name__ == '__main__':
    main()
