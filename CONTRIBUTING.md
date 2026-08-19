# 协作说明

这个仓库用于毕业设计实验协作。默认分支 main 保存可交接状态；同学不要直接在 main 上做大规模改动或启动长时间实验。

## 推荐流程

~~~powershell
git switch -c experiment/<short-name>
# 修改代码、配置或文档
python -m unittest tests\test_allocations.py
git add <explicit-files>
git commit -m "Describe the change"
git push -u origin experiment/<short-name>
~~~

然后在 GitHub 上创建 Pull Request，说明：

1. 改了什么、为什么改；
2. 使用了什么配置、数据划分和 seed；
3. 运行了哪些测试或验证；
4. 是否产生新的结果目录，以及结果是正式、开发还是诊断证据。

## 实验文件规则

- 不覆盖 results/*_v1 等已审计结果；新的协议使用新的 config 和 output directory。
- 不提交 *.safetensors、虚拟环境、缓存、访问令牌或本机路径配置。
- 长时间实验先在独立分支运行，完成后提交 checkpoint、配置和摘要；不要把未完成运行写成正式结论。
- 如果修改了固定路径，必须同步更新脚本、README 和对应文档，并运行验证脚本。
