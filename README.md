# SGC Project 2：机器学习收益预测（复现 Gu, Kelly & Xiu 2020）

团队：Sarah · Aria · Javier。目标是在 Matt 的 8 大基础因子（Momentum, Value, Quality, Yield,
Volatility, Size, Liquidity, Growth）下构建多个信号，用 GKX (2020) 的方法比较线性模型、树模型和神经网络
对个股下月超额收益的预测能力。

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # 单元测试
python scripts/run_demo.py                # 用模拟数据跑通全流程（约几分钟，CPU 即可）
python scripts/run_demo.py --feature-set factors --models ols ridge rf gbrt nn3   # 8 个因子合成指标版本
```

`run_demo.py` 默认用"快速模式"（小网格、小集成），只用来检查流程；正式结果加 `--full-grid`。

## 目录结构

```
p2ml/
  config.py            样本切分、发布滞后、股票池等所有设置
  data/
    crsp.py            CRSP 月度文件加载（CIZ msf_v2 为默认，SIZ 为备选）、清洗、构造超额收益和 h 月前瞻目标
    compustat.py       Compustat funda 加载、FF 账面价值、CCM 链接表、按 6 个月滞后合并
    free.py            免费数据：Ken French 因子、Welch-Goyal 宏观变量 → GKX 的 8 个宏观预测变量
    synthetic.py       与 CRSP/Compustat 字段完全一致的模拟数据（WRDS 批下来之前用）
  features/
    signals.py         各信号的计算（动量、波动、规模、流动性、估值、收益率、质量、成长）
    factors.py         FACTOR_MAP：8 大因子 → 信号 的映射，以及因子合成指标
    transform.py       GKX 截面排序标准化到 [-1,1]、中位数填充、宏观交互项、行业虚拟变量
  models/
    registry.py        模型库 + 超参数网格：ols, ols3, ols_huber, ridge, lasso, enet, pcr, pls, rf, gbrt, nn1-nn5
    nn.py              GKX 神经网络（32-16-8-4-2，BatchNorm、Adam、L1、早停、多种子集成）
  eval/
    splits.py          扩展训练窗 + 滚动验证窗 + 每年重新训练（默认与论文一致：1957-74 / 1975-86 / 1987-2016）
    metrics.py         样本外 R²（零基准）、修正 Diebold-Mariano 检验、rank IC、十分位多空组合与夏普
  pipeline.py          build_dataset()（原始数据 → 特征矩阵）和 run_backtest()（滚动训练、调参、预测）
scripts/run_demo.py    端到端示例（模拟数据）
scripts/run_wrds.py    真实数据：download 下载缓存，run 跑回测
tests/                 单元测试（标准化、切分、目标无前视、R² 定义、CIZ 与 SIZ 结果一致）
```

## 用真实 WRDS 数据运行

UofT 订阅了 CRSP 和 Compustat North America（没有 Compustat Global，本项目也用不到）。
CRSP 现在默认是 **CIZ（v2）格式**，代码直接读 `crsp.msf_v2`；如果查询失败会自动退回旧的 SIZ 表
`crsp.msf` + `crsp.msenames` + `crsp.msedelist`。CIZ 的 `mthret` 已经包含退市收益，不需要再合并退市文件。
CIZ 没有 `shrcd`/`exchcd`，代码用 WRDS 推荐的等价条件重建：普通股 = sharetype='NS'、securitytype='EQTY'、
securitysubtype='COM'、usincflg='Y'、issuertype in ('ACOR','CORP')；交易所 = primaryexch in N/A/Q，
且 conditionaltype='RW'、tradingstatusflg='A'。

```bash
pip install -r requirements.txt

# 第 1 步：从 WRDS 下载一次，缓存到 data/raw/*.parquet（会提示输入 WRDS 密码和 Duo 验证）
python scripts/run_wrds.py download --wrds-user miaoaria

# 第 2 步：先用小样本试跑（每月市值前 500 只、不加宏观交互项），几分钟内出结果
python scripts/run_wrds.py run --models ols3 ridge gbrt --top-n 500 --no-interactions

# 第 3 步：全样本 + 宏观交互项（需先从 Amit Goyal 主页下载 Welch-Goyal 的 xlsx 放到 data/raw/）
python scripts/run_wrds.py run --models ols3 enet pls rf gbrt nn3 --goyal-path data/raw/goyal.xlsx --full-grid
```

结果在 `outputs_wrds/`：`summary.csv`（样本外 R²、rank IC、多空夏普）、`tuning_log.csv`（每年每个模型的超参数）、
`predictions.parquet`（逐股逐月预测）。第一个检验：OLS-3 的月度 R² 应接近论文的约 0.16%。

默认测试期 1987–2016 与论文一致；`--last-test-year 2023` 可以延长。`--horizon 3/6/12` 换预测期限。

大数据量提示：CRSP 全样本约 300 万行 × 920 列，float32 约 11 GB。先用 `--top-n` 或 `--no-interactions`
或 `--feature-set factors` 跑通，再上全部交互项；NN 和 GBRT 建议在 GPU / 集群上跑。

## 已知简化（TODO）

- `retvol`、`maxret`、`ill` 目前用月度数据近似；GKX 用日度 CRSP（crsp.dsf），拿到后替换。
- 只用了年度 Compustat；季度信号（roaq、roeq 等，滞后 4 个月）待加。
- 信号数约 33 个，目标是每个因子 5-8 个、总计约 55 个；新信号加到 `signals.py` 并登记到 `FACTOR_MAP`。
- GLM（样条 + group lasso）尚未实现。
- 组合收益未扣交易成本；与 Project 1 的回测框架对接后补上。

## 参考

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial
Studies*, 33(5), 2223–2273. 免费版本：[NBER w25398](https://www.nber.org/papers/w25398.pdf)，
[SSRN 3159577](https://papers.ssrn.com/abstract=3159577)。
