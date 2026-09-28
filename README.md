# ModelScope Studio Collection

这个仓库集中整理我维护过的 ModelScope 创空间代码。每个创空间作为 `studios/` 下的独立目录保存，并补充模型来源、推理方案、关键技术、运行条件和在线体验链接。

仓库采用新的快照历史，不迁移各个创空间原仓库的 Git 提交记录。项目代码会逐个完成敏感信息、大文件和可运行性检查后再加入。

## 项目列表

| Studio | 任务 | 主要模型 | 推理技术 | ModelScope | 代码 | 状态 |
|---|---|---|---|---|---|---|
| LTX2.5 | 文生视频 / 图生视频 / 同步音频 | Lightricks/LTX-2.5 | DistilledPipeline · BF16 · CPU offload | [在线体验](https://modelscope.cn/studios/icebergyang/LTX2.5) | [`studios/LTX2.5`](studios/LTX2.5) | 已整理 |

更多创空间将逐个完成整理后加入。

## 目录结构

```text
modelscope_studio/
├── README.md
└── studios/
    └── <studio-name>/
        ├── README.md
        └── ...
```

## 使用说明

每个子项目有独立的依赖和运行要求。请进入对应目录阅读 README，不要假设所有创空间可以安装在同一个 Python 环境中。

模型权重通常不保存在本仓库，而是在首次运行时从 ModelScope 下载。ModelScope 创空间使用的 Secrets 和 `/mnt/workspace` 运行数据也不会复制到这里。
