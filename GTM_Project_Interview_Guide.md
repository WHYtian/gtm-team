# GTM Intelligence Multi-Agent System — 面试准备指南

> 目标岗位：AI Agent Project Intern @ BytePlus  
> 项目名称：GTM Intelligence Research Platform  
> 技术方向：ReAct Multi-Agent / RAG / Async Orchestration / LLM-as-Judge

---

## 一、项目概述

### 1.1 背景与目标

GTM（Go-To-Market，市场进入策略）研究是企业战略决策的核心环节，传统流程依赖人工查阅报告、整理竞品数据，耗时数天甚至数周。本项目构建了一个**全自动、多智能体协作的 GTM 情报研究平台**，用户只需输入一个市场主题（如"HR SaaS Market 2025"），系统在数分钟内自动完成：

- 多维度网络搜索与信息提取
- 用户上传行业报告与网络数据的交叉验证（RAG 融合）
- TAM/SAM/SOM、PESTEL、Porter's Five Forces 结构化分析
- 质量审查与多轮迭代修正
- 输出包含 Competitive Battle Cards 的完整 GTM Intelligence Report

### 1.2 核心指标

| 维度 | 数据 |
|------|------|
| 研究时长（原始人工） | 3-7 天 |
| 系统输出时长 | 5-12 分钟 |
| Agent 数量 | 6 个专职 Agent |
| 最大并发搜索 | 4 路并行 |
| RAG 知识库 | ChromaDB 持久化，all-MiniLM-L6-v2 嵌入 |
| 实时推流 | SSE (Server-Sent Events) 流式输出 |
| 部署方式 | 公网可访问（47.84.139.146:8091） |

---

## 二、系统架构

### 2.1 整体架构图

```
用户输入 (Topic)
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI + SSE 实时推流层                  │
└─────────────────────────────┬───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│              ReAct Supervisor (Doubao-Seed-2.0-Pro)      │
│   THINK → ACT 自由路由，带 6 层硬约束安全网                 │
└──┬──────┬──────┬──────┬──────┬──────────────────────────┘
   │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼
Researcher Synthesizer Analyst Critic Writer
(Alex)    (Jordan)   (Jamie) (Morgan)
   │                    │       │
   ▼                    ▼       ▼
Web Search +         TAM/SAM  Quality
Web Scrape           PESTEL   Review
+ RAG Query          Porter's (LLM-as-Judge)
                     Five Forces
                              │
                              ▼
                    ┌─────────────────┐
                    │ ChromaDB + RAG  │
                    │ LlamaIndex      │
                    │ all-MiniLM-L6   │
                    └─────────────────┘
                              │
                              ▼
                    GTM Intelligence Report
                    (Markdown + Battle Cards)
```

### 2.2 架构范式：ReAct（Reasoning + Acting）

系统核心采用 **ReAct 范式**，Supervisor 在每一轮输出：

```
THINK: [推理当前研究状态，判断下一步]
ACT: CALL_RESEARCHER | task: [具体指令]
```

**关键设计决策**：选择 ReAct 动态路由而非固定状态机（State Machine），原因：
- 固定流水线无法应对数据缺失、质量不达标等动态情况
- ReAct 让 Supervisor LLM 自行判断何时数据已充分、何时需要补充搜索
- 更接近人类研究员的真实决策流程

### 2.3 非 LangGraph 原生 Asyncio 实现

与业界常见的 LangGraph 框架不同，本项目使用 **Python 原生 asyncio** 构建，原因：
- 更精细的流控制（SSE 实时推流需要精确控制每条消息的时序）
- 避免 LangGraph 抽象层带来的调试困难
- 可以在 Agent 调用之间直接注入硬约束逻辑

---

## 三、Agent 设计与分工

### 3.1 六大 Agent 全景

| Agent | 名称 | 模型 | 职责 | 温度 |
|-------|------|------|------|------|
| Supervisor | S | Doubao-Seed-2.0-Pro | 全局路由决策，THINK→ACT | 0.3 |
| Researcher | Alex | Doubao-1.5-Pro-32K | 网络搜索、信息提取、RAG查询 | 0.3 |
| Synthesizer | Jordan | DeepSeek-V3 | Web 数据与 RAG 文档交叉验证 | 0.2 |
| Analyst | Jamie | DeepSeek-V3 | 结构化框架分析（TAM/PESTEL/Porter） | 0.5 |
| Critic | Morgan | DeepSeek-V3 | LLM-as-Judge 质量审查 | 0.6 |
| Writer | - | DeepSeek-V3 | 生成最终 GTM 报告 | 0.4 |

**模型分配策略**：通过 LLM-as-Judge 基准测试（benchmark）评估各模型在不同任务上的表现：
- Supervisor 使用高能力推理模型（Doubao-Seed）确保路由准确
- Researcher 使用长上下文模型（32K）处理网页内容
- Analyst/Critic 使用 DeepSeek-V3 平衡成本与分析质量

### 3.2 Supervisor 决策流程

```
每轮循环：
1. 构建 REACT_PROMPT（含 workspace 摘要、预算状态、已搜索内容）
2. Supervisor LLM 输出 THINK + ACT
3. 解析 ACT → action, param
4. 依次应用 6 层硬约束（Hard Constraints 0-5）
5. 执行 action → 更新 workspace
6. 检查终止条件（writer 完成 / MAX_ROUNDS 耗尽）
```

### 3.3 Researcher 两阶段预算（Dual Budget）

```
Pre-Analyst  Budget：3 次（初始4维并行 + 2次跟进补充）
Post-Analyst Budget：2 次（Critic 指出缺口后的定向补搜）
```

**设计意图**：防止无限循环搜索，同时保留 Critic 反馈后补充数据的能力。  
**配套规则**：`[RESEARCH: UNAVAILABLE]` 表示两次尝试均失败，禁止再次搜索同一指标（Anti-Loop Rule）。

### 3.4 Researcher 任务分解与并行执行

**初始搜索**：固定 4 个维度并行搜索：
1. Market Overview（市场规模、增速、预测）
2. Competitive Landscape（竞争格局、市占率）
3. Technology Trends（技术趋势、AI 影响）
4. Regulatory Environment（监管、合规）

**跟进搜索**：Supervisor 给定 directive → Researcher 自动分解为具体 query：
- Entity Coverage Rule：每个命名实体（公司/指标）生成恰好 1 条 query
- 最多 8 条 query，优先命名实体，市场级 query 排最后
- 以 4 路为一批并行执行，串行批次

**语义重试（Semantic Retry）**：
```python
if not _has_findings(summary):
    alt_q = researcher.suggest_alternative_query()
    retry_result = search(alt_q)
    if _has_findings(retry_result):
        use retry_result
    else:
        return [RESEARCH: UNAVAILABLE]
```

---

## 四、核心技术实现

### 4.1 硬约束安全网（Hard Constraints）

6 层硬约束在 Python 代码层面强制执行，不依赖 LLM 理解，确保路由正确性：

| 约束 | 触发条件 | 强制行为 |
|------|----------|----------|
| 0 | 首次 CALL_ANALYST 且用户有上传文档 | 先执行 Data Synthesizer |
| 1 | Researcher 预算耗尽 | 强制转 CALL_ANALYST |
| 2 | 准备写报告但 Analyst 后未经 Critic | 强制先 CALL_CRITIC |
| 3 | Analyst 已修订 ≥2 次 | 强制跳到 CALL_WRITER |
| 4 | Critic 给出 REJECT_DATA 且研究员尚未跟进 | 强制 CALL_RESEARCHER（后分析预算） |
| 5 | Critic 给出 logic_error 且 Supervisor 选了 CALL_RESEARCHER | 强制改为 CALL_ANALYST |

**为什么需要硬约束**：LLM 即便有明确 System Prompt 也无法 100% 遵守路由规则。硬约束作为确定性保障（deterministic guardrail），弥补 LLM 的随机性缺陷。

### 4.2 置信度标签体系（Confidence Tagging）

所有数据必须携带置信度标签，贯穿整个 pipeline：

```
[Data]       — 直接引用、有 URL 的数据点
[Estimate]   — 从相邻数据推算，必须列出公式
              例：TAM: ~$30B [Estimate — Gartner 2023 total HR software × 75% SaaS penetration]
[Claim]      — 厂商/营销材料声明，不可直接引用 URL
[Assumption] — Analyst 的策略性判断，必须说明依据
[N/A]        — 已确认不可获取
```

这套体系在 Researcher、Analyst、Critic、Writer 的 System Prompt 中共同约定，形成跨 Agent 的语义契约。

### 4.3 Data Synthesizer（Web × RAG 交叉验证）

**问题**：Researcher 搜索到网络数据，用户也上传了行业报告（PDF/CSV/Excel），两者可能互相印证或冲突，如何自动融合？

**方案**：Embedding-based Pre-filter + 单次 LLM 裁决

```
Step 1: 从 workspace 中提取所有 [Data/Estimate/Claim] 行 → N 条 findings
Step 2: 获取与当前 topic 相关的用户 RAG chunks → M 条 chunks
         （在初始研究阶段并行预取，namespace_filter='user'）
Step 3: 批量 embedding → cosine 相似度矩阵 (N × M)
         threshold = 0.42（经基准测试调优：真正匹配 100% recall，0 误触）
Step 4: 分类：
         max_sim ≥ 0.42 → OVERLAP PAIR（需 LLM 裁决）
         max_sim < 0.42 → RAG SUPPLEMENT（独立补充信息）
Step 5: 仅将 ~15 个匹配对 + supplements 送入单次 LLM 调用
```

**核心创新**：避免了 O(N×M) 次 LLM 调用（可能几百次），用 embedding 矩阵预筛选后只需 1 次 LLM 调用，大幅降低延迟和成本。

**裁决优先级规则**：
- 动态指标（市场规模、定价、营收）：以更新的年份为准
- 结构性指标（CAGR 方法论、细分定义）：分析师机构报告（Gartner/IDC）优先于厂商博客
- 范围差异（SaaS-only vs 全市场）：标注为 SCOPE MISMATCH，使用更窄范围的数字

### 4.4 RAG 系统设计

**技术栈**：LlamaIndex + ChromaDB + HuggingFace `all-MiniLM-L6-v2`

**Namespace 隔离**：
```
namespace = "user"     → 用户上传的行业报告
namespace = "platform" → 系统预置的参考数据（爬取内容）
```

**表格数据嵌入质量问题与解决**：

`all-MiniLM-L6-v2` 对 Markdown 表格嵌入质量差（管道符噪声 + 跨行截断），CSV/Excel 导入的数据相似度仅 0.26-0.32：

```python
# _tabular_to_sentences()：将 Markdown 表格转换为自然语言句子
# 原始格式（嵌入质量差）：
| 2025 | Global HR SaaS | $28.1B | Mordor Intelligence |
# 转换后（每行自成句子，可独立被理解）：
"Year: 2025. Market: Global HR SaaS. Size_Billion_USD: 28.1. Source: Mordor Intelligence."
```

转换后相似度提升到 0.33-0.35，显著改善召回率。

**去重策略**：`_dedup_document(filename, namespace)` 在重新上传同一文件时删除旧 chunks，避免重复索引。

### 4.5 语义缓存（Semantic Cache）

```
用户查询 → 向量化 → 与历史报告向量比较
  相似度 > 阈值（fresh）→ 直接返回缓存报告
  相似度 > 低阈值（stale）→ 携带历史报告作为背景运行新研究
  无匹配 → 全新研究
```

**意义**：防止同一市场被重复研究，缩短响应时间，同时允许用历史报告作为基底提升质量。

### 4.6 Workspace 上下文管理

Supervisor 在每轮看到的 workspace 内容需要精心设计，避免上下文爆炸：

```python
def _researcher_digest(output: str) -> str:
    """将 researcher 长输出压缩为关键发现摘要"""
    # 提取所有 [Data/Estimate/Claim] 行（最多10条）
    # 保留信号行 [RESEARCH: COMPLETE/WEAK/UNAVAILABLE]
    # Supervisor 始终看到核心数据，不会因截断遗漏关键指标
```

**设计权衡**：
- Researcher 原始输出可达 2000 字 → 压缩为 ~15 行关键发现
- Analyst/Critic 传给 Supervisor 截取前 700/1000 字（最近 3 轮显示完整内容）
- Analyst/Critic 接收全量数据（通过 `_build_ctx_for` 传 full output）

### 4.7 实时推流（SSE Streaming）

```python
async def emit(agent, content, phase, is_think=False):
    msg = {
        "type": "team_chat",
        "msg": {"agent": agent.agent_id, "content": content, 
                "ts": ..., "phase": phase, "is_think": is_think},
        "meta": {"name": agent.name, "color": agent.color, "avatar": agent.avatar},
    }
    await q.put(msg)  # asyncio.Queue → SSE 推送到前端
```

前端通过 `EventSource` 接收消息，实时渲染每个 Agent 的输出，用户可以观察到完整的推理过程。

---

## 五、关键创新点

### 5.1 动态 ReAct 路由 vs 固定 Pipeline

| 维度 | 固定 Pipeline | ReAct 动态路由（本项目） |
|------|--------------|------------------------|
| 灵活性 | 每轮固定调用顺序 | Supervisor 按需决策 |
| 数据缺口处理 | 只能继续 | 可补搜特定指标 |
| 质量控制 | 一次性 | Critic 反馈 → 迭代修正 |
| 成本优化 | 固定调用次数 | 数据充分即停止 |
| 可解释性 | 流程透明 | THINK 文本可审计 |

### 5.2 双预算机制（Dual Budget）

将 Researcher 的调用预算拆分为两个独立预算，解决了**前期探索 vs 后期定向填补**的资源分配问题：
- 不会因为初期搜索耗尽预算而无法响应 Critic 的缺口指摘
- 不会因为后期补搜预算独立存在而导致无限循环

### 5.3 Embedding 矩阵预筛选

Data Synthesizer 的 embedding pre-filter 是本项目的技术亮点之一：
- 将 NxM 次 LLM 比对降为 1 次 LLM 调用
- 纯 numpy 计算，无 API 调用，毫秒级完成
- 阈值 0.42 经过真实数据基准测试，平衡精度与召回

### 5.4 结构化 Critic 判决

Critic（Morgan）不输出自由文本，而是输出结构化判决：
```
[VERDICT: APPROVED]
[VERDICT: NEEDS_REVISION | reason: logic_error | issue: ...]
[VERDICT: NEEDS_REVISION | reason: missing_data | metric: ...]
[VERDICT: REJECT_DATA | claim: <具体数字> | search: <验证查询>]
```

Python 代码用正则解析判决类型 → 驱动硬约束路由，实现了 **LLM 输出驱动路由** 而非人工硬编码所有分支。

### 5.5 表格嵌入质量优化

`_tabular_to_sentences()` 将 CSV/Excel 的 Markdown 表格转换为自然语言句子，解决了结构化数据在 sentence-transformer 模型中嵌入质量差的问题。这是一个简洁但影响深远的工程决策。

---

## 六、技术难点与解决方案

### 难点 1：Supervisor 上下文盲区（Context Blindness）

**问题**：Supervisor 的 workspace 展示使用 `[:700]` 截断，导致 Researcher 第 4-8 条查询的结果被截断，Supervisor 误认为数据缺失，重复给出相同指令。

**解决**：`_researcher_digest()` 提取所有 `[Data/Estimate/Claim]` 标签行作为关键发现摘要，不受截断影响。Supervisor 始终看到所有关键数字。

### 难点 2：CSV/Excel 数据在 RAG 中召回率低

**问题**：Markdown 表格被 SentenceSplitter 截断，chunk 缺少列头信息，且管道符对 MiniLM 造成噪声，导致相似度 0.26-0.32（< 阈值 0.42）。

**解决**：`_tabular_to_sentences()` 在入库前将表格转换为自然语言句子，每行自包含完整语义，提升相似度到 0.33-0.35+。

### 难点 3：Asyncio + 同步 ChromaDB 阻塞事件循环

**问题**：ChromaDB 的 `col.get()` 是同步调用，直接在 async 函数中调用会阻塞事件循环，影响 SSE 推流和并发性能。

**解决**：
```python
result = await loop.run_in_executor(None, _get_user_rag_for_topic, topic)
```
将同步调用放入线程池执行，不阻塞事件循环。

### 难点 4：LLM 路由可靠性

**问题**：即便 System Prompt 中写明"logic_error → CALL_ANALYST"，Supervisor LLM 偶尔仍会选择 CALL_RESEARCHER。

**解决**：硬约束 4 和 5 在 Python 代码层面强制执行，LLM 的选择只是建议，最终决定由代码保障：
```python
if last_verdict == "REJECT_DATA" and not followed_up:
    if action != "CALL_RESEARCHER" and budget_available:
        action = "CALL_RESEARCHER"  # 强制覆盖
```

### 难点 5：Data Synthesizer 与主题无关的 RAG 噪声

**问题**：最初实现用 `_get_user_rag_chunks()` 获取所有用户上传文档，大量与当前主题无关的 chunks 进入对比，产生噪声。

**解决**：改为在初始研究时用 `query_rag_multi(..., namespace_filter='user')` 按 topic 预取相关 chunks，存储在 `user_rag_chunks` 变量中，Synthesizer 直接复用，既精准又避免额外 DB 调用。

### 难点 6：模型选型与成本控制

**问题**：不同 Agent 对能力的需求不同，全用高端模型成本爆炸，全用低端模型质量差。

**解决**：LLM-as-Judge 基准测试评估各模型在 Supervisor routing、数据提取、分析框架、质量审查等任务上的表现，按任务特性分配：
- 路由推理（Supervisor）→ 强推理模型（Doubao-Seed）
- 长文本处理（Researcher）→ 长上下文模型（32K）
- 分析/批评（Analyst/Critic）→ 平衡成本质量（DeepSeek-V3）

---

## 七、相关知识体系

### 7.1 Agent 核心范式

**ReAct (Reasoning + Acting)**
- 交替生成 Thought 和 Action，用推理指导行动
- 论文：`ReAct: Synergizing Reasoning and Acting in Language Models`
- 对比：CoT（Chain-of-Thought）只推理不行动，ReAct 同时具备两者

**Multi-Agent 系统拓扑**
- Sequential（本项目基础结构）：Agent 串行传递结果
- Parallel（本项目 Researcher 阶段）：多 Agent 并行处理不同子任务
- Supervisor-Worker（本项目核心）：Supervisor 动态分配任务给各 Worker Agent
- Hierarchical：多层 Supervisor 结构（本项目未采用，复杂度收益比低）

**LLM-as-Judge**
- 用 LLM 评估另一个 LLM 的输出质量
- 本项目：Critic（Morgan）评判 Analyst（Jamie）的分析质量
- 关键：Judge 需要与被评对象完全隔离，提供结构化判决格式

### 7.2 RAG（Retrieval-Augmented Generation）

**基本流程**：Document → Chunking → Embedding → VectorStore → Query → Retrieve → Augment → Generate

**Chunking 策略对比**：
- Fixed-size：简单但可能截断语义
- Sentence Splitter（本项目）：按语义边界切分
- Hierarchical：保留父子关系（LlamaIndex 支持）

**Embedding 模型**：
- `all-MiniLM-L6-v2`：轻量（80MB），速度快，本地推理，适合工程原型
- `text-embedding-3-large`：OpenAI，质量更高但有 API 延迟和成本
- BGE 系列：中文场景更优

**Similarity 度量**：
- Cosine Similarity：与向量长度无关，最常用
- Dot Product：需要归一化向量
- 本项目阈值 0.42 是经验值，通过 26 对真实数据基准测试确定

### 7.3 Context Engineering

**Prompt 设计原则**：
- 角色定义（Role）：明确 Agent 的身份、能力边界
- 输出格式约束（Format）：结构化 template 减少幻觉
- 示例（Few-shot）：正确 vs 错误示例对比
- 负向约束（Guard-rail）：明确禁止行为（如不得发明数字）

**系统提示分层**：
- REACT_SYSTEM（Supervisor）：全局策略、预算规则、路由逻辑
- Agent System Prompt（各 Worker）：角色专职化、输出 template

**Workspace 上下文管理**：
- 问题：随对话轮次增加，context 膨胀 → 成本上升，LLM 注意力分散
- 方案：digest 压缩（关键发现摘要）+ 差异化截断（新 vs 旧条目不同截断长度）

### 7.4 工程技术栈

| 技术 | 用途 | 知识点 |
|------|------|--------|
| FastAPI | Web 框架 | async endpoint, SSE |
| asyncio | 并发 | event loop, gather, run_in_executor |
| ChromaDB | 向量数据库 | persistent client, metadata filter |
| LlamaIndex | RAG 框架 | VectorStoreIndex, SentenceSplitter, StorageContext |
| HuggingFace | 本地 Embedding | sentence-transformers, 离线部署 |
| numpy | 矩阵计算 | cosine similarity matrix |
| pypdf / openpyxl | 文档解析 | 多格式支持 |

### 7.5 评估指标

**RAG 质量**：
- Precision@K：Top-K 检索结果中相关文档比例
- Recall@K：所有相关文档中被检索到的比例
- NDCG：考虑排序的综合指标

**Agent 质量**（本项目通过 Critic 实现）：
- 数据准确性（有无 Citation）
- 逻辑一致性（推理是否自洽）
- 覆盖完整性（框架要素是否齐全）

**Embedding 阈值调优**：
- 制作 ground-truth 测试集（MATCH/UNRELATED/RELATED 三类）
- 在不同阈值下计算 Precision/Recall，选择 F1 最优点

---

## 八、可能被问到的问题 & 答案

### Q1：为什么选择 ReAct 而不是固定流水线？

**答**：固定流水线适合任务确定、输入规范的场景，但 GTM 研究天然是探索性的——搜索结果的质量不可预测，数据缺口事先未知，分析框架需要动态填补空白。ReAct 的 Supervisor 可以在每一步基于当前状态（workspace 内容、预算余量、质量信号）做出最优决策。例如：当 Researcher 发现某指标为 `[RESEARCH: UNAVAILABLE]` 时，Supervisor 可以立即让 Analyst 用 `[N/A]` 处理，而不是浪费一次预算在已知无法找到的数据上。固定流水线无法做到这种弹性。

---

### Q2：你如何防止 Agent 之间的"幻觉传播"（Hallucination Propagation）？

**答**：采用三层防御：
1. **置信度标签**：每条数据必须携带 `[Data/Estimate/Claim/Assumption]`，禁止裸数字，追踪数据来源和可信度
2. **Plausibility Check**：Researcher 在报告任何市场规模前，必须自我检验量级是否合理（SaaS 子市场通常 $5B-$80B，超过 $100B 必须标注怀疑）
3. **Critic 的 REJECT_DATA 机制**：发现特定数字疑似错误时，触发定向重新搜索验证，而非被动接受

---

### Q3：Data Synthesizer 的 embedding 阈值 0.42 是怎么确定的？

**答**：构建了 26 对基准测试数据，包含 MATCH（真正相关）、RELATED（同领域但不同指标）、UNRELATED（完全不相关）三类。在不同阈值下测试：
- 阈值过低（如 0.3）：大量不相关内容通过，LLM 调用成本爆炸且结果噪声多
- 阈值 0.42：真正 MATCH 对达到 100% recall，CMO 行业无关文档（最高相似度 0.366）完全被过滤
- 阈值过高（如 0.5）：部分真实相关对被漏掉

最终选 0.42 作为工程上的最优平衡点。

---

### Q4：你的多 Agent 系统如何处理成本控制问题？

**答**：多层次成本控制：
1. **预算上限**：Researcher 最多 3+2=5 次调用，Analyst 最多修订 2 次（硬约束）
2. **语义缓存**：相似主题直接命中缓存，不重复调用
3. **Embedding 预筛选**：Data Synthesizer 用纯 numpy 矩阵计算代替 N×M 次 LLM 调用
4. **模型差异化**：Supervisor 用强推理模型，Analyst/Critic 用性价比模型
5. **Workspace 压缩**：Supervisor 看 digest 而非原始长文，减少 prompt token 数
6. **反循环规则**：已标 UNAVAILABLE 的指标不再重搜

---

### Q5：如何保证 Researcher 并行搜索后 Supervisor 能看到所有发现？

**答**：这是项目中遇到的实际 Bug。最初 Supervisor 的 workspace 对 Researcher 输出只取前 700 字符，导致第 4-8 条查询的结果完全不可见，Supervisor 误认为数据缺失，重复给出相同搜索指令。

解决方案是 `_researcher_digest()`：从原始输出中正则提取所有 `[Data/Estimate/Claim]` 标签行，无论原始输出多长，Supervisor 始终能看到所有关键数字。同时保留信号行（`[RESEARCH: COMPLETE/WEAK/UNAVAILABLE]`），让 Supervisor 判断数据充分程度。

---

### Q6：如果 Critic 给出 REJECT_DATA，整个系统如何响应？

**答**：
1. Critic 输出结构化判决：`[VERDICT: REJECT_DATA | claim: "$5T global SaaS 2025" | search: "global SaaS market size 2025"]`
2. 硬约束 4（Python 代码层）检测到最近 Critic 判决为 REJECT_DATA 且此后未有 Researcher 跟进
3. 无论 Supervisor LLM 选择什么，强制将 action 改为 `CALL_RESEARCHER`，并从 Critic 判决中提取 `search` 字段作为查询指令
4. Researcher 搜索替代数据源
5. 若后分析预算耗尽，硬约束 1 进一步将 CALL_RESEARCHER 改为 CALL_ANALYST，Analyst 将该指标标注为 `[Assumption]` 或 `[N/A]`

---

### Q7：这个系统和 Dify / Coze 这类 Low-Code Agent 平台有什么区别？

**答**：
- **控制粒度**：Low-code 平台提供预设节点，复杂的条件路由（如依据 Critic 判决类型分叉）很难实现；本项目可以精确控制每个 Agent 调用的上下文、预算、约束
- **实时性**：SSE 流式推流需要精确的 asyncio 控制，Low-code 平台通常只支持一次性返回
- **可调试性**：自写代码可以在 Agent 调用之间插入日志、断点、状态检查；Low-code 的黑盒调试困难
- **适用场景**：Low-code 适合快速原型验证简单流程；High-code 适合有复杂业务逻辑、需要精细成本控制、需要定制集成的生产系统

---

### Q8：你在这个项目中最大的技术挑战是什么？

**答**：最具挑战的是 **Supervisor 路由可靠性** 与 **上下文管理** 的协同设计。

LLM Supervisor 的核心矛盾是：给它太少信息 → 决策失误；给它太多信息 → context 过长、注意力分散、成本上升。解决方案是：Supervisor 只看结构化摘要（digest），但 Analyst/Critic 看完整原始数据，两层视图服务不同目的。

另外，LLM 即便有详细的路由规则也不能 100% 遵守，必须用确定性的硬约束兜底。最初只依赖 System Prompt 描述路由规则，发现实际运行中 REJECT_DATA 后 Supervisor 仍会有概率直接调 Analyst。改为代码层面强制后，路由正确性才达到 100%。

---

### Q9：如果要扩展这个系统，你会怎么做？

**答**：
1. **Memory / 长期记忆**：当前语义缓存只保存完整报告，可以扩展为细粒度的"市场洞察知识库"，让 Researcher 在搜索前先查自建知识库
2. **SFT/RL 微调 Supervisor**：用高质量对话数据（THINK+ACT 对）微调 Supervisor，让路由准确性从硬约束兜底变为模型内化
3. **用户反馈闭环**：用户对报告的评分/编辑操作作为 reward signal，通过 RLHF 持续优化各 Agent
4. **更丰富的工具（Tool Use）**：接入 Bloomberg、Crunchbase 等付费 API，扩展数据源
5. **多模态支持**：处理图表、幻灯片（PPT）中的数据，扩展文档导入能力

---

### Q10：你了解 SFT / RL 微调吗？如何应用到本项目？

**答**：了解基本概念。
- **SFT（Supervised Fine-Tuning）**：用标注好的 (input, ideal_output) 对监督训练模型。应用到本项目：收集 Supervisor 做出正确 THINK→ACT 决策的对话数据（包括正确处理 logic_error、REJECT_DATA 的案例），用这些数据微调 Supervisor，减少对硬约束的依赖。
- **RL（强化学习）**：通过 reward signal 引导模型输出。应用场景：以 Critic 的 APPROVED/NEEDS_REVISION 作为 reward，训练 Analyst 输出更高质量的分析，减少被 Critic 打回的次数。
- **CT（Continual Training / 持续训练）**：在基础模型上持续注入领域数据，保持知识新鲜度。应用场景：定期用最新 GTM 报告数据更新 Researcher 的知识边界。

目前项目中没有微调，使用 prompt engineering + 硬约束来弥补，但 SFT 是进一步提升路由准确性和分析质量的自然演进路径。

---

## 九、项目亮点话术（面试开场介绍）

> "我构建了一个基于 ReAct 范式的 GTM 情报多智能体系统，包含 6 个专职 Agent，通过 Supervisor 动态路由实现自动化市场研究。系统核心创新在于三点：第一，用 embedding 矩阵预筛选代替 O(N×M) 次 LLM 调用来融合 RAG 与网络数据；第二，用结构化 Critic 判决驱动路由逻辑，并用 Python 硬约束作为确定性安全网弥补 LLM 路由的随机性；第三，设计双预算机制分离探索阶段与定向填补阶段的 Researcher 调用，防止资源浪费和无限循环。整个系统已部署上线，用户可以在 5-12 分钟内获取原本需要数天人工整理的 GTM 报告。"

---

*文档生成日期：2026-05-01*  
*项目仓库：/home/admin/gtm-team/*
