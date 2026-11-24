# 工业场景自适应模型自进化系统

DriftGuard-IM 是系统子名称。

本仓库发布项目自有 Harness、漂移监测实现和真实本地实验记录。公开数据 A/B/C 各完成 10 轮确定性续训。实验记录在 experiments/reproduced，输入分布审计在 experiments/drift。模拟协议测试明确使用 mock call ID，不冒充真实平台调用。

平台模型选择器已核验 Qwen/Qwen3.8-27B。正式平台试运行证据将在实际运行后补充，当前没有正式成绩。

## 复现

Python 3.13，NumPy 2.2.4。使用赛事发布的 CPU baseline 和数据，将 baselines 放在 src/baselines，release 下保持 cpu-a、cpu-b、cpu-c 目录。官方代码独立再分发授权不明，因此本公开仓库不重新发布它们；评审完整源码包按赛事通道交付。

SDK 1.0.4 下载：https://rsi-index.115.zone/downloads/contest-harness-sdk-1.0.4.tar.gz
SHA-256: 7afd282d6b5fd889d64d0515a67d169806970e36200689b18d2040026167a4af
解压至 src/sdk，使 src/sdk/contest_harness 存在。

```sh
PYTHONPATH=src:src/sdk python3 src/run_experiments.py --release release --output reproduced
PYTHONPATH=src:src/sdk python3 src/run_drift_audit.py --release release --output drift
DRIFTGUARD_RELEASE=release PYTHONPATH=src:src/sdk python3 -m unittest discover -s src -p 'test_*.py' -v
```

本地实验重复使用公开验证分区，结果不代表独立隐藏测试泛化。没有新增人工标签；历史确定性实验没有大模型 token 消耗。平台版本每轮调用模型规划与决策，并保存真实 call ID、usage 和宿主指标。未提供 usage 时记为 null。

新增代码及文档经过 AI 辅助整理与测试。源码不包含账户密钥和隐藏测试数据。第三方来源见 src/THIRD_PARTY_NOTICES_CN.md。
