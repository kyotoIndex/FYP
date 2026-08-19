# 数据目录

项目当前保存已准备好的本地数据，避免正式运行时依赖临时下载。

- SST-2/：主实验使用的二分类数据。
- QNLI/：句对分类数据。
- AG_NEWS/：四分类新闻数据。
- raw/：原始下载或转换输入；QNLIv2.zip 保留作 provenance。
- DATASET_MANIFEST.json：数据文件和统计清单。

配置使用 data/ 下的 TSV。数据预处理和外部数据准备入口见 scripts/prepare_external_datasets.py。
