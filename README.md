# AutoRec — Automotive Recommendation System

基于协同过滤与内容特征的汽车推荐系统，涵盖合成数据生成、ETL 清洗、特征工程、
离线评估与在线 API 服务，支持 Popularity / Content-Based / ALS 三种推荐算法。

---

## 目录结构

```
AutoRec/
├── autorec/
│   ├── etl/
│   │   ├── synthetic_generator.py   # 合成用户 / 车辆 / 交互数据
│   │   ├── transform.py             # 清洗函数
│   │   ├── quality_checks.py        # 数据质检 + 完整 ETL 入口
│   │   └── load.py                  # 写入 SQLite / PostgreSQL
│   ├── db/
│   │   ├── connection.py            # SQLAlchemy 引擎 & Session
│   │   └── schema.py                # ORM 模型（4 张表）
│   ├── features/
│   │   ├── user_features.py         # 用户特征（聚合统计）
│   │   └── item_features.py         # 车辆特征（数值 + OHE + TF-IDF）
│   ├── models/
│   │   ├── base.py                  # BaseRecommender 抽象类
│   │   ├── popularity.py            # 流行度推荐
│   │   ├── content_based.py         # 内容协同过滤（余弦相似度）
│   │   └── matrix_factorization.py  # 隐式 ALS（implicit 库）
│   ├── eval/
│   │   ├── metrics.py               # P@K / R@K / HR@K / NDCG@K / Coverage / Diversity
│   │   └── evaluator.py             # 时序切分 + 全量离线评估
│   └── api/
│       └── main.py                  # FastAPI 推理接口（5 个端点）
├── data/
│   ├── raw/                         # 原始 CSV（git 忽略）
│   ├── processed/                   # 清洗后 CSV + 评估结果（git 忽略）
│   └── features/                    # Parquet 特征矩阵（git 忽略）
├── sql/
│   ├── 01_create_schema.sql         # 星型 Schema DDL
│   └── 02_create_indexes.sql        # 5 种索引
├── scripts/
│   └── run_all.py                   # 一键复现脚本（步骤 1-6）
├── tests/                           # 单元 / 集成测试
├── ui/
│   └── dashboard.py                 # Streamlit 三页仪表板
├── .env.example
├── requirements.txt
└── README.md
```

---

## 快速开始（一键复现）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2-6. 自动执行数据生成 → ETL → 特征工程 → 评估
python scripts/run_all.py
```

完成后直接启动服务：

```bash
# 终端 A：启动 API（端口 8000）
python autorec/api/main.py

# 终端 B：启动 Dashboard（端口 8501）
streamlit run ui/dashboard.py
```

打开浏览器访问 http://localhost:8501

---

## 分步运行

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 生成合成数据

```bash
python -m autorec.etl.synthetic_generator
```

输出到 `data/raw/`：`users.csv`（5000 行）、`cars.csv`（3000 行）、`interactions.csv`（~50000 行）

### 3. ETL 清洗 + 质检

```bash
python -m autorec.etl.quality_checks
```

执行清洗（价格过滤、去重、外键校验）并打印每列通过/失败报告，结果保存到 `data/processed/`。

### 4. 加载数据库

```bash
python -m autorec.etl.load
```

默认写入 `data/autorec.db`（SQLite）。切换 PostgreSQL：编辑 `autorec/etl/load.py`，
将 `USE_SQLITE = False`，并在 `.env` 中填写连接信息。

```bash
# PostgreSQL 初始化（可选）
psql -d autorec -f sql/01_create_schema.sql
psql -d autorec -f sql/02_create_indexes.sql
```

### 5. 特征工程

```bash
# 用户特征（聚合统计 + 多样性熵）
python -m autorec.features.user_features

# 车辆特征（StandardScaler + OneHotEncoder + TF-IDF）
python -m autorec.features.item_features
```

输出到 `data/features/`：`user_features.parquet`、`item_features.parquet`

### 6. 离线评估

```bash
python -m autorec.eval.evaluator
```

时序切分（最后 20% 为测试集），对三个模型计算 Recall@10 / NDCG@10 / HitRate@10 /
Coverage / Diversity，结果保存到 `data/processed/eval_results.csv`。

### 7. 启动 API 服务

```bash
python autorec/api/main.py
# 或
uvicorn autorec.api.main:app --host 0.0.0.0 --port 8000 --reload
```

| 端点 | 说明 |
|---|---|
| `GET  /health` | 服务状态 + 已加载模型列表 |
| `GET  /cars/{car_id}` | 车辆详情 |
| `GET  /users/{user_id}/profile` | 用户特征 |
| `POST /recommend` | 获取推荐（支持 price / body_type 过滤） |
| `GET  /metrics` | 读取离线评估指标 |

交互式文档：http://localhost:8000/docs

### 8. 启动 Dashboard

```bash
streamlit run ui/dashboard.py
```

访问 http://localhost:8501，包含三个页面：

- **User Recommendations**：输入用户 ID，选择模型，查看推荐结果与解释
- **Model Comparison**：Recall / NDCG / Coverage 柱状图 + 冷启动热力图
- **Data Overview**：价格分布、品牌交互量、用户活跃度、购买转化率

> Dashboard 在 API 离线时自动降级到本地模型，不影响使用。

---

## 配置（PostgreSQL）

```bash
cp .env.example .env
```

编辑 `.env`：

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=autorec
DB_USER=postgres
DB_PASSWORD=your_password
```

---

## 模型说明

| 模型 | 算法 | 冷启动 | 特点 |
|---|---|---|---|
| Popularity | 加权计数（purchase×5, test_drive×1）| 全局排名 | 基线，Coverage 极低 |
| ContentBased | 余弦相似度（item 特征矩阵）| 无法推荐 | Coverage 最高（98%） |
| ALS | Implicit Feedback ALS（factors=64）| Popularity fallback | 精度最优 |

---

## 评估结果（参考）

| Model | Recall@10 | NDCG@10 | HitRate@10 | Coverage | Diversity |
|---|---|---|---|---|---|
| Popularity | 0.0034 | 0.0022 | 0.0084 | 0.004 | 0.769 |
| ContentBased | 0.0042 | 0.0025 | 0.0096 | **0.984** | 0.272 |
| ALS | **0.0047** | **0.0029** | **0.0104** | 0.414 | 0.757 |
