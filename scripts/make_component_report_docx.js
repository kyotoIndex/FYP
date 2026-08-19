/* Create a Chinese Word briefing from audited result files and generated figures. */
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, AlignmentType, HeadingLevel, ImageRun, BorderStyle, PageNumber, Footer } = require('docx');

const ROOT = path.resolve(__dirname, '..');
const REPORT = path.join(ROOT, 'reports', 'pcla_component_report');
const FIGURES = path.join(REPORT, 'figures');
const METHODS = ['fedhello', 'fedra_random_partial', 'pcla_component_alternating', 'random_component_alternating'];
const LABELS = {
  fedhello: 'Fed-HeLLo',
  fedra_random_partial: 'Random partial layer',
  pcla_component_alternating: 'PCLA component',
  random_component_alternating: 'Random component',
};

function csvRows(file) {
  const [header, ...lines] = fs.readFileSync(file, 'utf8').trim().split(/\r?\n/);
  const names = header.split(',');
  return lines.map((line) => Object.fromEntries(line.split(',').map((value, index) => [names[index], value])));
}

function chinese(text, options = {}) {
  return new TextRun({ text, font: 'Microsoft YaHei', ...options });
}

function cell(text, bold = false) {
  return new TableCell({ children: [new Paragraph({ children: [chinese(text, { bold, size: 19 })] })] });
}

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [chinese(text, { bold: true })] });
}

function figure(file, caption) {
  const data = fs.readFileSync(path.join(FIGURES, file));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ data, transformation: { width: 570, height: 326 } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [chinese(caption, { italics: true, size: 18 })] }),
  ];
}

const metrics = new Map(csvRows(path.join(REPORT, 'report_metrics.csv')).map((row) => [row.method_label, row]));
const borders = { top: { style: BorderStyle.SINGLE, size: 4, color: '808080' }, bottom: { style: BorderStyle.SINGLE, size: 4, color: '808080' }, left: { style: BorderStyle.SINGLE, size: 4, color: '808080' }, right: { style: BorderStyle.SINGLE, size: 4, color: '808080' }, insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'BFBFBF' }, insideVertical: { style: BorderStyle.SINGLE, size: 2, color: 'BFBFBF' } };

const summaryRows = [
  new TableRow({ children: ['方法', '最终准确率', '末 5 轮均值', '总时间（秒）', '总通信（MB）'].map((x) => cell(x, true)) }),
  ...METHODS.map((method) => {
    const r = metrics.get(LABELS[method]);
    return new TableRow({ children: [
      cell(LABELS[method]),
      cell(Number(r.final_accuracy_mean).toFixed(4)),
      cell(Number(r.last5_accuracy_mean).toFixed(4)),
      cell(Number(r.final_time_mean).toFixed(2)),
      cell(Number(r.final_comm_mb_mean).toFixed(2)),
    ] });
  }),
];

const doc = new Document({
  styles: { default: { document: { run: { font: 'Microsoft YaHei', size: 21 } } } },
  sections: [{
    properties: { page: { margin: { top: 900, bottom: 900, left: 1050, right: 1050 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [chinese('PCLA 联邦 LoRA 实验汇报 | 页 ', { size: 16 }), new TextRun({ children: [PageNumber.CURRENT], size: 16 })] })] }) },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [chinese('PCLA 联邦 LoRA 组件分配实验汇报', { bold: true, size: 38 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 }, children: [chinese('修正标签分布后的四算法对比（SST-2 / BERT-base）', { size: 24 })] }),
      heading('1. 汇报目的'),
      new Paragraph({ children: [chinese('比较四种联邦 LoRA 训练策略在准确率、后期稳定性、模拟训练时间和通信量方面的表现，并明确当前 PCLA component 方法可以成立和不能成立的结论。')] }),
      heading('2. 实验设置'),
      new Paragraph({ children: [chinese('数据集为 SST-2，基础模型为 bert-base-uncased。实验使用 5 个客户端、每客户端 512 条训练样本、12 轮联邦训练、3 个随机种子（7、13、21）。客户端数据使用受约束 Dirichlet 标签划分，保证每个种子的联邦训练集合整体标签比例与原始训练集一致。')] }),
      new Paragraph({ children: [chinese('低、中、高资源客户端分别可训练 6、9、12 个 LoRA 层等价预算。时间和通信量为基于测得本地操作时间与设定设备/网络系数的模拟指标，并非真实多设备墙钟测试。')] }),
      heading('3. 四种算法'),
      new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, borders, rows: [
        new TableRow({ children: ['算法', '简要机制'].map((x) => cell(x, true)) }),
        new TableRow({ children: [cell('Fed-HeLLo'), cell('先进行 RGD warm start，之后用服务器 proxy 数据估计 FIM 重要性，按客户端资源能力分配完整 LoRA 层。')] }),
        new TableRow({ children: [cell('Random partial layer'), cell('每个客户端按资源容量随机选择完整 LoRA 层，训练该层的完整 LoRA 参数。')] }),
        new TableRow({ children: [cell('Random component'), cell('在匹配的稀疏 component 预算下随机选择 rank components；奇偶轮交替训练 LoRA-B 与 LoRA-A。')] }),
        new TableRow({ children: [cell('PCLA component'), cell('在相同 component 预算下，根据客户端本地数据得到的 component 重要性、EMA、探索与新鲜度分数选择 components；同样交替训练 A/B。')] }),
      ] }),
      heading('4. 汇总结果'),
      new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, borders, rows: summaryRows }),
      ...figure('01_accuracy_trajectory.png', '图 1：三 seed 平均准确率轨迹；阴影为样本标准差。'),
      ...figure('02_accuracy_summary.png', '图 2：最终轮准确率与末 5 轮平均准确率。'),
      ...figure('03_efficiency.png', '图 3：平均累计模拟时间与通信量。'),
      ...figure('04_accuracy_communication_tradeoff.png', '图 4：末 5 轮准确率与累计通信量的权衡。'),
      heading('5. 可辩护结论'),
      new Paragraph({ children: [chinese('（1）在完全相同的稀疏 component 预算下，PCLA component 的最终准确率为 0.8231，而 Random component 为 0.7736，提升约 4.95 个百分点；末 5 轮均值也由 0.7691 提升至 0.8336。')] }),
      new Paragraph({ children: [chinese('（2）PCLA component 的通信量为 44.58 MB，较 Random partial layer 和 Fed-HeLLo 的 176.20 MB 减少约 74.7%。')] }),
      new Paragraph({ children: [chinese('（3）PCLA component 尚未在最终准确率上超过 Fed-HeLLo（0.8418）或 Random partial layer（0.8321）。因此应将结果表述为：在固定稀疏组件预算下，重要性引导的客户端特定 component allocation 优于匹配的随机 component 对照，并以显著通信节省接近层级基线。')] }),
      heading('6. 报告边界与下一步'),
      new Paragraph({ children: [chinese('本实验仅覆盖 SST-2、BERT-base、5 客户端、12 轮和 3 个种子；不能据此声称对不同模型、数据集或大规模客户端设置具有普遍优势。当前 PCLA component 的资源约束主要通过容量预算体现，component 选择分数尚未显式包含计算/通信成本。下一步应完成 20 客户端、30 轮、修正划分的正式多 seed 实验，并增加 LoRA-A² 风格选择策略与 component-level cost-aware 选择的消融。')] }),
      heading('7. 结果来源'),
      new Paragraph({ children: [chinese('原始审计结果：results/pcla_component_constrained_dirichlet_v1。所有图表均从该目录的 per-seed metrics、aggregate_summary.csv 与 derived_accuracy_summary.csv 自动生成。')] }),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = path.join(REPORT, 'PCLA_component_four_method_briefing.docx');
  fs.writeFileSync(out, buffer);
  console.log(`Wrote ${out}`);
});
