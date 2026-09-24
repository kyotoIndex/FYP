# 发布材料核验

本目录只整理已经完成的实验；核验没有启动新的训练。

在本目录运行 `PYTHONDONTWRITEBYTECODE=1 python3 reproduce/verify_publication.py` 通过：39 个来源文件 SHA-256、1,750 条已有端点、148 项原配对比较、22 项聚合诊断比较、补充批次 10 条逐种子端点和 300 轮、225 条正式训练曲线与 11,600 行逐轮读数，以及 115 条成本作业、4,500 轮和 23 个五种子条件。补充批次 seed 13 自第 21 轮起的单类预测计入五种子均值。

在 `reproduce/executed_sources/` 运行 `PYTHONDONTWRITEBYTECODE=1 python3 verify_public.py` 通过：441 个发布文件、129 项冻结队列绑定的源码 SHA-256、313 条执行回执记录的归档哈希。第三方上游代码按固定 commit 和四个执行文件哈希引用；模型权重、原数据集和完整运行 ZIP 不进入本目录。

`qpdf --check data/curves/complete_round_curves.pdf` 通过。主 README、方法说明、来源表及各附表 README 的相对链接均存在；公开目录内没有个人设备或 HPC 用户的绝对路径，也没有扫描到常见私钥头和 GitHub token 格式。公开路径脱敏副本与原执行字节的 SHA 对应关系由各自的清单保留。

成本表中的服务器 SVD 耗时和 EMA 额外评价耗时为“原运行未单独记录”。SVD 作业的高下载计数包括重复发送完整基础模型，按实际记录保留。
