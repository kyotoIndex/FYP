"""Generate the result tables of the V4.1 + S1 report directly from the analysis outputs."""
import csv
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
MAIN = RUN / 'attempt2/analysis'
EXTRA = RUN / 'analysis_extra_main_s1'
S1 = RUN / 's1/analysis/audits'
LABEL = {
    'central_pooled_both': '集中式：本轮 cohort 样本合并，A/B 同时训练，优化器跨轮保持',
    'central_iid_both': 'S1 集中式：每轮从全体 12800 条 IID 抽同样数量，配方同 central_pooled_both',
    'central_serial_both': '集中式：按客户端块串行，每块重置优化器，A/B 同时训练',
    'central_serial_alt': '集中式：按客户端块串行，每块重置优化器，A/B 交替',
    'fed_both_mean': '联邦：A/B 同时训练，因子分别平均（FedIT 式）',
    'fed_both_exact': '联邦：同上，服务器补回乘积残差（FedEx-LoRA 式）',
    'fed_ffa_b': 'S1 联邦：A 冻结、只训 B（FFA-LoRA 式）',
    'old_full_alt': '联邦：全部 rank，A/B 交替（v3 full_alt，复用）',
    'old_pcla': 'PCLA，固定分母 F（v3，复用）',
    'old_replay_random_rank': '固定层与配额、随机 rank，F（v3，复用）',
    'old_random': '随机组件，F（v3，复用）',
    'online_pcla_contributors': 'PCLA 在线选择 + 贡献者分母 C',
    'online_replay_contributors': '同 donor 层与配额的随机 rank + C',
    'trace_pcla_contributors': 'PCLA 旧支持序列 + C',
    'trace_pcla_function_norm_matched': 'PCLA 旧支持序列 + G',
    'trace_replay_random_rank_contributors': 'replay 旧支持序列 + C',
    'trace_replay_random_rank_function_norm_matched': 'replay 旧支持序列 + G',
    'trace_random_contributors': 'random 旧支持序列 + C',
    'trace_random_function_norm_matched': 'random 旧支持序列 + G',
}


def f(x, n=1):
    return '—' if x is None else f'{x:.{n}f}'


def main():
    rows = list(csv.DictReader((EXTRA / 'trajectory_readouts.csv').open()))
    cells = defaultdict(list)
    for r in rows:
        cells[r['cell']].append(r)
    endpoints = {(re.sub(r'_s\d+$', '', r['job']), int(r['seed'])): 100 * float(r['accuracy'])
                 for r in csv.DictReader((MAIN / 'per_seed_endpoints.csv').open())}
    for path in glob.glob(str(S1 / '*.json')):
        job = Path(path).stem
        if job.startswith('smoke'):
            continue
        rep = json.load(open(path))
        endpoints[(re.sub(r'_s\d+$', '', job), rep['seed'])] = 100 * statistics.mean(
            x['accuracy'] for x in rep['rows'] if 26 <= x['round'] <= 30)
    out = ['# 结果表（由 build_result_tables.py 生成）', '',
           '主端点：每 seed 第 26–30 轮 benchmark dev（872 条）平均 accuracy，%。旧 v3 轨迹复用前经 parity 逐位核对。', '',
           '## 表 1　各格子端点与轨迹读数（5 seed）', '',
           '| 格子 | 做法 | 均值 | seed 7 / 13 / 21 / 42 / 87 | 峰值−末五轮 (pp) | 第11–30轮单类预测轮数 | cohort 正类比例与预测正类比例相关 | 末五轮 logits 平均后 |',
           '|---|---|---:|---|---:|---:|---:|---:|']
    order = ['central_iid_both', 'central_pooled_both', 'central_serial_both', 'central_serial_alt',
             'fed_both_mean', 'fed_both_exact', 'fed_ffa_b', 'old_full_alt', 'old_pcla', 'old_replay_random_rank',
             'old_random', 'online_pcla_contributors', 'online_replay_contributors', 'trace_pcla_contributors',
             'trace_pcla_function_norm_matched', 'trace_replay_random_rank_contributors',
             'trace_replay_random_rank_function_norm_matched', 'trace_random_contributors',
             'trace_random_function_norm_matched']
    for c in order:
        rs = cells[c]
        per = [endpoints[(c, s)] for s in (7, 13, 21, 42, 87)]
        # Pre-registered readouts; old v3 trajectories have no saved logits.
        m = lambda k: statistics.mean(float(x[k]) for x in rs) if rs and rs[0].get(k) not in ('', None) else None
        logit = m('last5_logit_average_accuracy')
        out.append(f"| `{c}` | {LABEL[c]} | {statistics.mean(per):.2f} | {' / '.join(f'{v:.1f}' for v in per)} | "
                   f"{f(m('peak_minus_last5_pp'))} | {f(m('near_single_class_rounds_11_30'))} | "
                   f"{f(m('corr_cohort_vs_predicted_positive_11_30'), 2)} | {f(None if logit is None else 100 * logit)} |")
    out += ['', '## 表 2　v4.1 预定比较（配对差，pp；描述性 t 95% 区间，df=4）', '',
            '| 比较 | 五 seed | 区间 | seed 7/13 | seed 21/42/87 |', '|---|---:|---|---:|---:|']
    groups = {g: json.load(open(MAIN / f'{g}.json')) for g in
              ('all_five', 'prior_LR_development', 'not_used_for_prior_LR_ranking')}
    for a, b, c in zip(*groups.values()):
        ci = a.get('descriptive_t_95_pp')
        out.append(f"| {a['comparison']} | {a['mean_difference_pp']:+.2f} | "
                   f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | {b['mean_difference_pp']:+.2f} | {c['mean_difference_pp']:+.2f} |")
    out += ['', '## 表 3　执行前登记的补充比较（EXECUTION.md 5.2）', '',
            '| 比较 | 五 seed | 区间 | 各 seed |', '|---|---:|---|---|']
    extra = json.load(open(EXTRA / 'extra_comparisons.json'))['all_five']
    for r in extra:
        ci = r.get('descriptive_t_95_pp')
        out.append(f"| {r['comparison']} | {r['mean_difference_pp']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                   + ' / '.join(f"{v:+.1f}" for v in r['per_seed_pp'].values()) + ' |')
    out += ['', '## 表 4　G0 同状态干预（第 5/15 轮父状态后两轮，train 内 tuning 集，5 seed×2 锚点×2 轮=20 格）', '',
            '| 支持 | 规则 | accuracy | balanced accuracy | NLL |', '|---|---|---:|---:|---:|']
    g0 = defaultdict(list)
    for r in csv.DictReader((MAIN / 'fixed_state_metrics.csv').open()):
        g0[(r['support'], r['rule'])].append(r)
    for (sup, rule), rs in sorted(g0.items()):
        out.append(f"| {sup} | {rule} | {100 * statistics.mean(float(x['accuracy']) for x in rs):.2f} | "
                   f"{100 * statistics.mean(float(x['balanced_accuracy']) for x in rs):.2f} | "
                   f"{statistics.mean(float(x['nll_float64']) for x in rs):.3f} |")
    target = RUN / 'RESULT_TABLES.md'
    target.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
