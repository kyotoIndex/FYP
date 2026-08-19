# 模型文件

这里保留可公开、体积较小的 tokenizer 和 config 文件，帮助确认模型结构。大模型权重不提交到 GitHub：

- models/roberta-base/model.safetensors 约 499 MB，已被 .gitignore 排除。
- BERT-base 由配置中的 local_files_only=true 控制，需要在运行机器的 Hugging Face 缓存中准备。

因此，克隆仓库后还需要按本机硬件和模型缓存补齐权重；仅凭 GitHub 仓库不能保证立即启动训练。
