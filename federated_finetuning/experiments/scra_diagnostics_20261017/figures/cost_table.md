# 运行成本：峰值显存与单次运行时间

由 `make_figures.py` 生成。数据：d0 运行取 `d0_rounds.json` 顶层的 `peak_memory_mib` / `elapsed_seconds`，strong-frac sweep 的 r8 / r2 / 前缀异构取 `result.json` 的 `cost` 字段。所有运行都是 150 轮、每轮 10 个客户端、每客户端 20 步本地训练；时间是整次运行的墙钟时间（含每 10 轮一次的 50 客户端评估）。运行分布在多个 RTX 4090 D 节点上（见“节点”列），时间还受节点负载影响，同一方法不同 seed 之间就相差可达 10% 以上，因此不宜用这些秒数比较方法的计算量。

## 按方法汇总

| 方法 | 运行数 | 峰值显存 MiB | 秒 / 次：均值（最小–最大） |
|---|---:|---:|---|
| r8（全员 rank 8） | 3 | 1557 | 733（723–742） |
| r2（全员 rank 2） | 3 | 1557 | 748（737–757） |
| 前缀异构 L0 | 3 | 1557 | 741（734–749） |
| 前缀异构，20% 强端 | 3 | 1557 | 741（736–744） |
| vref（完整训练，上传 r/R） | 3 | 1557 | 758（741–786） |
| vref，r = 1（budget8） | 1 | 1557 | 812 |
| vref，rank 混合 | 1 | 1557 | 736 |
| vref，20% 强端 | 1 | 1557 | 756 |
| L2（受限训练） | 3 | 1557 | 780（756–827） |
| L2，r = 1 | 1 | 1557 | 763 |
| L2，rank 混合 | 1 | 1557 | 811 |
| L2，20% 强端 | 1 | 1557 | 787 |
| RCS 在线收缩 | 1 | 1557 | 824 |
| 固定收缩 0.63 | 1 | 1557 | 841 |
| 弱端学习率 × 0.6 | 1 | 1557 | 814 |
| amp（放大受限训练） | 3 | 1558 | 851（817–909） |
| amp，α = 2 | 1 | 1558 | 858 |
| amp，α = 8 | 1 | 1558 | 821 |
| amp，拥有者平均 | 1 | 1558 | 876 |
| amp，前缀分配 | 1 | 1558 | 817 |
| amp，r = 1 | 1 | 1558 | 875 |
| amp，rank 混合 | 1 | 1558 | 856 |
| amp，20% 强端 | 1 | 1558 | 805 |
| A1a FSLoRA 前向 + amp 梯度路径 | 3 | 1557 | 874（869–878） |
| A1b FSLoRA 前向 + Hájek | 3 | 1557 | 815（769–840） |
| A1c FSLoRA 前向 + 零填充 | 3 | 1557 | 793（771–830） |
| A2 学习率 × R/r，服务器 ÷ R/r | 3 | 1557 | 804（795–811） |
| FSLoRA 式（外部） | 3 | 1557 | 812（805–822） |
| Fed-PLoRA 式（外部） | 3 | 1557 | 794（791–797） |
| FLASC 式（外部） | 3 | 1557 | 768（758–782） |

全部 58 次运行的峰值显存在 1556–1558 MiB 之间：在 RoBERTa-base 规模下，受限训练与放大对峰值显存几乎没有影响，资源动机需要在大模型上另行论证（roadmap 阶段 C）。上传量没有在 d0 运行中记录，未列入此表。

## 逐次运行

| 运行 | 来源 | 峰值显存 MiB | 秒 | 节点 |
|---|---|---:|---:|---|
| r8_s2040 | strong-frac sweep | 1557 | 742 | hpcgpu13 |
| r8_s2041 | strong-frac sweep | 1557 | 735 | hpcgpu13 |
| r8_s2042 | strong-frac sweep | 1556 | 723 | hpcgpu13 |
| r2_s2040 | strong-frac sweep | 1557 | 757 | hpcgpu13 |
| r2_s2041 | strong-frac sweep | 1557 | 737 | hpcgpu13 |
| r2_s2042 | strong-frac sweep | 1556 | 749 | hpcgpu13 |
| prefix_s2040 | strong-frac sweep | 1557 | 734 | hpcgpu13 |
| prefix_s2041 | strong-frac sweep | 1557 | 749 | hpcgpu13 |
| prefix_s2042 | strong-frac sweep | 1556 | 740 | hpcgpu13 |
| prefix_sf20_s2040 | strong-frac sweep | 1557 | 744 | hpcgpu13 |
| prefix_sf20_s2041 | strong-frac sweep | 1557 | 736 | hpcgpu13 |
| prefix_sf20_s2042 | strong-frac sweep | 1556 | 743 | hpcgpu13 |
| vref_s2040 | diag v1_vref | 1557 | 741 | hpcgpu13 |
| vref_s2041 | batch 1 | 1557 | 786 | hpcgpu13 |
| vref_s2042 | batch 1 | 1556 | 746 | hpcgpu15 |
| vref_r1_s2040 | diag v1_budget8 | 1557 | 812 | hpcgpu14 |
| vref_het_s2040 | batch 1 | 1557 | 736 | hpcgpu14 |
| vref_sf20_s2040 | batch 1 | 1557 | 756 | hpcgpu13 |
| L2_s2040 | diag v1 | 1557 | 827 | hpcgpu13 |
| L2_s2041 | batch 1 | 1557 | 758 | hpcgpu14 |
| L2_s2042 | batch 1 | 1556 | 756 | hpcgpu14 |
| L2_r1_s2040 | batch 1 | 1557 | 763 | hpcgpu14 |
| L2_het_s2040 | batch 1 | 1557 | 811 | hpcgpu13 |
| L2_sf20_s2040 | batch 1 | 1557 | 787 | hpcgpu14 |
| rcs_s2040 | diag v1_rcs | 1557 | 824 | hpcgpu13 |
| fixedshrink_s2040 | diag v1_fixedshrink | 1557 | 841 | hpcgpu13 |
| weaklr_s2040 | diag v1_weaklr | 1557 | 814 | hpcgpu11 |
| amp_s2040 | batch 1 | 1558 | 828 | hpcgpu14 |
| amp_s2041 | batch 1 | 1558 | 817 | hpcgpu14 |
| amp_s2042 | batch 1 | 1558 | 909 | hpcgpu13 |
| amp_a2_s2040 | batch 1 | 1558 | 858 | hpcgpu13 |
| amp_a8_s2040 | batch 1 | 1558 | 821 | hpcgpu15 |
| amp_owner_s2040 | batch 1 | 1558 | 876 | hpcgpu13 |
| amp_prefix_s2040 | batch 1 | 1558 | 817 | hpcgpu15 |
| amp_r1_s2040 | batch 1 | 1558 | 875 | hpcgpu13 |
| amp_het_s2040 | batch 1 | 1558 | 856 | hpcgpu13 |
| amp_sf20_s2040 | batch 1 | 1558 | 805 | hpcgpu14 |
| fslora_st_s2040 | batch 2 | 1556 | 878 | hpcgpu14 |
| fslora_st_s2041 | batch 2 | 1557 | 875 | hpcgpu14 |
| fslora_st_s2042 | batch 2 | 1556 | 869 | hpcgpu14 |
| fslora_hajek_s2040 | batch 2 | 1557 | 840 | hpcgpu14 |
| fslora_hajek_s2041 | batch 2 | 1557 | 769 | hpcgpu13 |
| fslora_hajek_s2042 | batch 2 | 1556 | 836 | hpcgpu14 |
| fslora_zerofill_s2040 | batch 2 | 1557 | 779 | hpcgpu13 |
| fslora_zerofill_s2041 | batch 2 | 1557 | 830 | hpcgpu14 |
| fslora_zerofill_s2042 | batch 2 | 1556 | 771 | hpcgpu13 |
| lrcomp_s2040 | batch 2 | 1557 | 811 | hpcgpu14 |
| lrcomp_s2041 | batch 2 | 1557 | 795 | hpcgpu14 |
| lrcomp_s2042 | batch 2 | 1556 | 807 | hpcgpu14 |
| fslora_paper_s2040 | batch 2 | 1557 | 805 | hpcgpu14 |
| fslora_paper_s2041 | batch 2 | 1557 | 822 | hpcgpu14 |
| fslora_paper_s2042 | batch 2 | 1556 | 809 | hpcgpu14 |
| fedplora_s2040 | batch 2 | 1557 | 797 | hpcgpu14 |
| fedplora_s2041 | batch 2 | 1557 | 794 | hpcgpu14 |
| fedplora_s2042 | batch 2 | 1556 | 791 | hpcgpu14 |
| flasc_s2040 | batch 2 | 1557 | 782 | hpcgpu14 |
| flasc_s2041 | batch 2 | 1557 | 765 | hpcgpu14 |
| flasc_s2042 | batch 2 | 1556 | 758 | hpcgpu14 |
