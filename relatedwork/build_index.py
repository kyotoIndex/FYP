"""Generate a Chinese navigation page from the download manifest."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
records=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
lines=['# FYP 相关论文与原始 LaTeX 源码',
 '', '检索与下载日期：2026-09-30（香港时间）。已下载 20 篇论文 PDF，其中 11 篇特别接近的论文附有作者在 arXiv 公开的原始 TeX 工程。',
 '', '范围：联邦 LoRA 微调、客户端资源与数据异构、层/秩/组件预算分配、因子聚合和个性化。结合当前 `pcla_experiments/docs/HETSA_LORA_V1.md` 与 PCLA/PCLA-WL 项目文档筛选；同时覆盖两条研究线。',
 '', '筛选深度：论文官网的元数据与摘要，未逐篇完整审读，未复现论文实验。下列相关性是用于安排阅读顺序的判断，不是认定方法等价、实验可直接比较或创新性已得到确认。2026 年新预印本的结果也应经过进一步审读。',
 '', '## 建议阅读顺序',
 '', '1. HetSA：FedSA-LoRA → Fed-HeLLo / LEGEND → FedAS-LoRA → FedSAIL → FDLoRA。先厘清共享 A、本地 B 的既有基础，再检查异构层预算与个性化的贡献边界。',
 '2. PCLA/PCLA-WL：Fed-HeLLo → Fed-pilot → LEGEND → HAFLQ → Fed-PLoRA → AdaLoRA。重点核对分配单位、重要性定义、真实成本与异构聚合。',
 '3. A/B 训练与聚合：RoLoRA → AS-LoRA → FFA-LoRA → FLoRA / FlexLoRA。注意 AS-LoRA 中的 component 指 A/B 因子，不是 PCLA 的 rank-1 组件。',
 '', '## 文件组织',
 '', '- `01_closest/`：11 篇特别接近的论文 PDF。',
 '- `02_related/`：5 篇强相关论文 PDF。',
 '- `03_foundations/`：4 篇基础论文 PDF。',
 '- `tex/方法_arXiv编号版本/`：作者原始 `.tex` 及配套图、`.bib`、模板；保留作者文件名。',
 '- `source_archives/`：arXiv 原始 source 响应备份，文件扩展名 `.source`；实际封装可能为 gzip/tar 或单个 TeX。',
 '- `references.bib`：20 篇的 BibTeX，按 arXiv 元数据生成；正式投稿前按目标刊会核对发表信息。',
 '- `paper_index.csv`：可检索索引。`manifest.json`：作者、固定版本、来源 URL、摘要、文件路径、SHA-256 和下载状态。',
 '- `verification.json`：PDF 解析及文件完整性检查。',
 '', '## 论文索引',
 '', '| 方法 | 相关程度 | 年份（初次提交） | 与 FYP 的关系 | 本地 PDF | 官网 |',
 '|---|---|---|---|---|---|']
for r in records:
 lines.append('| '+r['method']+' | '+r['priority']+' | '+r['date'][:4]+' | '+r['relation_zh']+' | [PDF]('+r['pdf_file']+') | [arXiv]('+r['url']+') |')
lines += ['', '## 特别接近论文的 TeX 主文件', '', '这里提供的是作者公开的原始源码，不是 PDF 转换出的近似文本，也不是我重写的论文。下载版 PDF 和源码固定为同一个 arXiv 版本。',
 '', '| 方法 | 论文标题 | 主文件入口 | TeX 文件数 |', '|---|---|---|---|']
for r in records:
 if r['tex_status']=='ok':
  mains=r.get('tex_main_candidates',[]) or r['tex_files'][:1]
  links=' / '.join('['+Path(p).name+']('+p+')' for p in mains)
  lines.append('| '+r['method']+' | '+r['title']+' | '+links+' | '+str(len(r['tex_files']))+' |')
lines += ['', '主文件入口由 `\\documentclass` 和 `\\begin{document}` 检查确定。源码仅下载和解包，未编译；是否可直接编译还取决于多文件依赖、作者模板、字体和 TeX 环境。',
 '', '## 引用时要核对的区别',
 '', '- FedSA-LoRA 已提出共享 A、保留本地 B；我们不能把这一点作为首次提出的贡献。HetSA 应对照异构层预算下的本地效果和实际成本。',
 '- HAFLQ 包含重要性感知截断/冻结与 rank-1 矩阵聚合；它与组件级 PCLA 很接近，应优先读方法部分，不能只将它视为量化工作。',
 '- 层数、秩数、逻辑 mask、实际计算开销是不同量。对照论文时需区分训练时间、峰值显存与上传/下载字节。',
 '- 本地个性化准确率与单个共享服务器模型准确率需分别报告，不能互换。',
 '- 项目代码里的统一 rank `fedlora` 基线是受控实现名称，不能仅凭名称认定为某篇 FedLoRA/FedIT 论文的完整复现。',
 '', '## 检索中发现但尚未纳入下载的候选',
 '', '- **HRALoRA**：Heterogeneous-Resource-Aware Federated Learning with Intelligent LoRA Allocation and Aggregation。检索命中 CCGrid 2026，DOI [10.1109/CCGrid68966.2026.00043](https://doi.org/10.1109/CCGrid68966.2026.00043)。未取得可验证的公开 PDF 下载地址及作者 TeX，因此不计入已下载的 20 篇。',
 '', '## 旧目录中的引用编号问题',
 '', '`F:/FYP/references/papers/FedIT_2303.10512v2.pdf` 的文件名把 FedIT 与 AdaLoRA 的 arXiv 编号混在一起。官网确认 `2303.10512` 对应 AdaLoRA，FedIT 对应 `2305.05644`。本目录分别下载两篇正确编号的论文；旧文件未改动，旧 PDF 的实际正文身份未在本次核对。',
 '', '## 可重复下载',
 '', '在安装了 requests 的 Python 环境中运行 `python F:/FYP/relatedwork/download_papers.py`。该脚本从官网读取版本并下载，对不同内容的已有文件拒绝覆盖。原始 source 只安全解包普通文件，不执行任何作者代码。再运行 `python F:/FYP/relatedwork/build_index.py` 更新此索引。',
 '']
(ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
print('README generated; entries:',len(records))
