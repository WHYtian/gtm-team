# GitHub GTM Research Agent 开源项目深度调研报告

> 调研日期：2026-04-30  
> 关键词：GTM research agent, market research AI agent, competitive intelligence agent, multi-agent market analysis, go-to-market AI, sales intelligence agent  
> 调研范围：最近6个月内有更新，优先 star 数较多项目

---

## 目录

1. [TradingAgents（56.4k stars）](#1-tradingagents)
2. [GPT Researcher（26.8k stars）](#2-gpt-researcher)
3. [dzhng/deep-research（18.8k stars）](#3-dzhng-deep-research)
4. [langchain-ai/open_deep_research（11.3k stars）](#4-langchain-aiopen_deep_research)
5. [guy-hartstein/company-research-agent（1.7k stars）](#5-guy-hartstein-company-research-agent)
6. [ferdinandobons/startup-skill（261 stars）](#6-ferdinandobons-startup-skill)
7. [MaxKmet/idea-validation-agents（93 stars）](#7-maxkmet-idea-validation-agents)
8. [chadboyda/agent-gtm-skills（36 stars）](#8-chadboyda-agent-gtm-skills)
9. [psrane8/Market-Research-Agent（22 stars）](#9-psrane8-market-research-agent)
10. [SalmaSalahEldin/Multi-Agent-Market-Research（7 stars）](#10-salmasalaheldin-multi-agent-market-research)
11. [ramamoorthy07/Multi-Agent-Market-Research-and-Use-Case（5 stars）](#11-ramamoorthy07-multi-agent-market-research-and-use-case)
12. [Nirikshan95/VettIQ（13 stars）](#12-nirikshan95-vettiq)
13. [CrewAI 官方示例 - Marketing Strategy Crew](#13-crewai-官方示例---marketing-strategy-crew)
14. [与当前 gtm-team 项目的综合对比](#14-与当前-gtm-team-项目的综合对比)

---

## 1. TradingAgents

**链接**：https://github.com/TauricResearch/TradingAgents  
**Stars**：56,400  
**最后更新**：2026-04-25（v0.2.4）  
**语言**：Python 99.8%  
**License**：Apache-2.0  
**领域定位**：金融交易，但架构模式高度可迁移到 GTM research

### 整体架构

虽然是交易领域，但 TradingAgents 提供了业界最成熟的"多 agent 分析 + 辩论 + 决策"范式，对 GTM research 有极高参考价值。

```
                         ┌─────────────────────────────┐
                         │      INPUT: 目标股票 + 日期    │
                         └─────────────┬───────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │              Analyst Team（并行）                  │
              │  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────┐  │
              │  │Fundament-│ │Sentiment │ │  News  │ │Tech- │  │
              │  │als Analyst│ │ Analyst  │ │Analyst │ │nical │  │
              │  └──────────┘ └──────────┘ └────────┘ └──────┘  │
              └────────────────────────┬────────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │           Researcher Team（Bull vs Bear 辩论）     │
              │     ┌─────────────────┐  ┌────────────────────┐ │
              │     │  Bull Researcher │  │  Bear Researcher   │ │
              │     └─────────────────┘  └────────────────────┘ │
              └────────────────────────┬────────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │              Trader Agent（决策合成）               │
              └────────────────────────┬────────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │      Risk Management + Portfolio Manager          │
              └────────────────────────┬────────────────────────┘
                                       │
                         ┌─────────────▼───────────────┐
                         │    交易信号 + 执行报告          │
                         └─────────────────────────────┘
```

**框架**：LangGraph（状态图 + 检查点恢复）  
**协作模式**：Parallel（Analyst Team）→ Sequential（辩论）→ Sequential（决策）

### Agent 角色划分

| Agent | 角色 | 职责 |
|-------|------|------|
| Fundamentals Analyst | 基本面分析师 | 分析财务报表、公司画像、财务历史 |
| Sentiment Analyst | 情绪分析师 | 分析社媒情绪，使用评分算法 |
| News Analyst | 新闻分析师 | 监控全球新闻和宏观经济指标 |
| Technical Analyst | 技术分析师 | 使用 MACD、RSI 等指标检测价格模式 |
| Bull Researcher | 多头研究员 | 构建做多证据链，反驳空头论点 |
| Bear Researcher | 空头研究员 | 构建做空证据链，反驳多头论点 |
| Trader | 交易员 | 综合所有报告做出交易决策 |
| Risk Manager + Portfolio Manager | 风险管理 | 评估波动性和流动性风险，审批/拒绝交易提案 |

### 关键 System Prompt（原文引用）

**News Analyst：**
```
"You are a news researcher tasked with analyzing recent news and trends over the past week. 
Please write a comprehensive report of the current state of the world that is relevant for 
trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) 
for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, 
limit) for broader macroeconomic news. Provide specific, actionable insights with supporting 
evidence to help traders make informed decisions. Make sure to append a Markdown table at the 
end of the report to organize key points in the report, organized and easy to read."
```

**Fundamentals Analyst：**
```
"analyze fundamental information over a company...write comprehensive reports covering 
financial documents, company profiles, and financial history...provide specific, actionable 
insights with supporting evidence...include a Markdown summary table."
```

**Bull Researcher：**
```
"You are a Bull Analyst advocating for investing in the stock. Your task is to build a 
strong, evidence-based case emphasizing growth potential, competitive advantages, and 
positive market indicators."
```

**Bear Researcher：**
```
"You are a Bear Analyst making the case against investing in the stock. Your goal is to 
present a well-reasoned argument emphasizing risks, challenges, and negative indicators."
```

### Tools 列表

| Agent | Tools |
|-------|-------|
| Fundamentals Analyst | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` |
| News Analyst | `get_news(query, start_date, end_date)`, `get_global_news(curr_date, look_back_days, limit)` |
| Sentiment Analyst | `get_news`, 情绪评分算法 |
| Technical Analyst | `get_stock_data`, `get_indicators`（MACD, RSI） |

### 使用模型
OpenAI、Anthropic、Google、xAI、DeepSeek、Qwen、GLM、OpenRouter、Ollama、Azure OpenAI（全支持，可配置）

### 输出格式
- 各 Analyst 生成 Markdown 报告（含表格）
- Bull/Bear 辩论记录
- 最终交易信号 + 反思日志（Decision Log with Performance Reflection）
- Checkpoint 恢复机制

### 与 gtm-team 对比亮点
- **Bull/Bear 辩论机制**：两个对立 researcher 就同一主题辩论，最终由 Judge 或 Trader 裁决。可直接迁移为 GTM 场景下的"机会派 vs 风险派"辩论，生成更平衡的市场分析。
- **LangGraph + SqliteSaver**：支持 per-ticker 检查点恢复，中断后继续。gtm-team 可借鉴实现长任务断点续跑。
- **多模型并行调用**：不同 agent 使用不同模型（如 Flash 用于大量文本处理，高级模型用于最终决策）。

---

## 2. GPT Researcher

**链接**：https://github.com/assafelovic/gpt-researcher  
**Stars**：26,800  
**最后更新**：2026-04-16（v3.4.4）  
**语言**：Python 55.2%, TypeScript 27.6%  
**贡献者**：257+  
**定位**：第一个开源深度研究 agent，可用于任意话题的 web 和本地文档研究

### 整体架构

双模式：单 Agent（GPTResearcher 核心）+ 多 Agent（LangGraph 编排的 Chief Editor 团队）

```
多 Agent 模式（STORM 论文启发）：

  Human Input (query)
       │
  ┌────▼─────┐
  │ Browser  │  → 初始网页浏览（研究问题收集）
  └────┬─────┘
  ┌────▼─────┐
  │ Planner  │  → 规划文章章节结构（基于初始研究）
  │ (Editor) │    输出: JSON {title, date, sections[]}
  └────┬─────┘
       │  ← Optional: Human Feedback
  ┌────▼──────────────────────────────────────────────────┐
  │         Parallel Research (每个章节一个子任务)            │
  │  ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
  │  │ResearchAgent│    │ReviewerAgent│    │ReviserAgt │ │
  │  │ (GPTResrch) │──► │ (审核质量)    │──► │ (修订)    │ │
  │  └─────────────┘    └─────────────┘    └───────────┘ │
  └───────────────────────────────┬───────────────────────┘
  ┌─────────────────────          │
  │  WriterAgent                  │
  │  (生成 intro/conclusion/TOC)   │
  └─────────────────────┬─────────┘
  ┌─────────────────────▼─────────┐
  │  PublisherAgent               │
  │  (输出 PDF/DOCX/Markdown)      │
  └───────────────────────────────┘
```

**框架**：LangGraph（StateGraph）+ 可选 AG2  
**协作模式**：Sequential（Planning）→ Parallel（Per-section Research）→ Conditional（Review Loop: Accept or Revise）→ Sequential（Write + Publish）

### Agent 角色划分

| Agent | 颜色标识 | 职责 |
|-------|---------|------|
| MASTER / ChiefEditor | 淡黄 | 总协调：生成任务ID，初始化团队，编译LangGraph工作流 |
| RESEARCHER | 浅蓝 | 执行 GPTResearcher，进行初始研究和按章节深度研究 |
| EDITOR | 黄 | 规划文章章节，管理并行研究，维护 Review 循环 |
| WRITER | 浅绿 | 基于研究生成 intro/conclusion/TOC，输出带引用的 Markdown |
| REVIEWER | 青 | 审核草稿是否符合 guidelines，给出"通过/修订"反馈 |
| REVISOR | 白 | 根据 Reviewer 反馈修订内容 |
| PUBLISHER | 洋红 | 生成最终报告（PDF/DOCX/Markdown），管理文件存储 |
| Human | — | 可选人工反馈节点（Human-in-the-Loop） |

### 关键 System Prompt（原文引用）

**EditorAgent（规划 prompt）：**
```
"You are a research editor. Your goal is to oversee the research project from inception 
to completion. Your main task is to plan the article section layout based on an initial 
research summary."

指令：生成最多 {max_sections} 个章节标题（不含 introduction/conclusion/references），
返回 JSON: {title, date, sections: []}
```

**WriterAgent：**
```
"You are a research writer. Your sole purpose is to write a well-written research reports 
about a topic based on research findings and information."

主要写作 prompt 要求：生成详细的 introduction 和 conclusion，含 markdown 超链接引用，
返回 JSON: {table_of_contents, introduction, conclusion, sources}
```

**ReviewerAgent：**
```
"You are an expert research article reviewer. Your goal is to review research drafts and 
provide feedback to the reviser only based on specific guidelines."

逻辑：草稿满足所有 guidelines 返回 None；否则返回具体修订建议。
修订稿审核：'ONLY if critical since the reviser has already made changes'
```

### Tools 列表
- GPTResearcher（核心）：Tavily、Bing、Google、DuckDuckGo 等多种搜索引擎（可配置）
- Web scraping（JavaScript 支持）
- 本地文档分析（PDF、Excel、Word、Markdown、PowerPoint）
- MCP 服务器集成
- 图片抓取 + AI 生成插图（Google Gemini）

### 使用模型
- 默认：OpenAI o3-mini（Deep Research 模式）
- 支持任意 LLM 提供商（OpenRouter、Ollama 等）
- 摘要/压缩使用可配置的轻量模型

### 输出格式
- 5-6 页研究报告
- 多格式：PDF、DOCX、Markdown
- 含引用表格、ToC、完整参考文献

### 与 gtm-team 对比亮点
- **Review-Revise 循环**：Reviewer → Revisor 形成自动质量保障闭环，避免人工校对。gtm-team 目前缺少内置的报告质量自检机制。
- **章节级并行研究**：按规划的章节结构并行分配研究任务，比按角色分工更灵活。
- **Human-in-the-Loop**：在规划阶段插入人工反馈节点，支持用户调整研究方向。

---

## 3. dzhng/deep-research

**链接**：https://github.com/dzhng/deep-research  
**Stars**：18,800  
**最后更新**：活跃维护中（77 commits）  
**语言**：TypeScript 97.6%（Node.js）  
**定位**：最简实现（<500行代码）的迭代式深度研究 agent，强调可读性和可复刻性

### 整体架构

单 Agent 递归架构（无角色分工），通过广度/深度参数控制研究范围：

```
User Query + {breadth, depth}
       │
  ┌────▼──────────────────────────────┐
  │  generateSerpQueries()            │
  │  输出: breadth 个搜索查询            │
  └────┬──────────────────────────────┘
       │  (并发限制: pLimit=2)
  ┌────▼──────────────────────────────┐
  │  For each query:                  │
  │  firecrawl.search() → 5 results  │
  │  processSerpResult()              │
  │  → learnings[], followUpQuestions │
  └────┬──────────────────────────────┘
       │
  ┌────▼──────────────────────────────┐
  │  if depth > 0:                    │
  │    deepResearch(                  │
  │      newBreadth = ceil(breadth/2) │
  │      newDepth = depth - 1        │
  │      accumulated learnings...    │
  │    )  ← 递归调用                  │
  └────┬──────────────────────────────┘
       │
  ┌────▼──────────────────────────────┐
  │  writeFinalReport()               │
  │  输出: report.md 或 answer.md      │
  └───────────────────────────────────┘
```

**协作模式**：单 Agent 递归（广度减半，深度递减），无 Supervisor

### Agent 角色划分
无角色分工，单一 LLM 承担所有任务，通过不同 prompt 切换行为：
- **Query Generator**：生成搜索查询
- **Learning Extractor**：从搜索结果提取 learnings
- **Follow-up Generator**：识别需要继续研究的方向
- **Report Writer**：汇总所有 learnings 生成最终报告

### 关键 Prompt（原文引用）

**Follow-up Questions（研究开始前）：**
```
"Generate follow-up questions to clarify the research intent"
```

**SERP Query Generation：**
```
"generate a list of SERP queries...Return a maximum of [numQueries]...
Make sure each query is unique"
```

**Learning Extraction：**
```
"generate a list of learnings from the contents...Return a maximum of [numLearnings]...
include any entities like people, places, companies"
```

**Final Report：**
```
"write a final report...aim for 3 or more pages, include ALL the learnings from research"
```

### Tools 列表
- **Firecrawl API**：搜索 + 内容提取（唯一外部工具依赖）
- 支持 OpenRouter 兼容端点

### 使用模型
- 默认：OpenAI o3-mini
- 可配置：DeepSeek R1（via Fireworks）、本地端点
- 理由选项：低/中/高（影响 token 消耗）

### 输出格式
- `report.md`（完整研究报告，3+ 页）
- `answer.md`（直接简洁回答）

### 与 gtm-team 对比亮点
- **极简实现**：不到 500 行 TypeScript，是理解递归研究 agent 的最佳参考。研究广度随深度自动减半（breadth/2），避免指数级 API 调用。
- **无结构化 Agent 角色**：对比 gtm-team 的多 agent 分工，此方案用单 agent + 不同 prompt 完成全流程，适合快速原型验证。
- **Firecrawl 集成**：相比 Tavily，Firecrawl 支持更完整的页面内容提取（JavaScript 渲染）。

---

## 4. langchain-ai/open_deep_research

**链接**：https://github.com/langchain-ai/open_deep_research  
**Stars**：11,300  
**最后更新**：2025-08-14  
**语言**：Python  
**Forks**：1,600  
**定位**：LangChain 官方深度研究 agent，Deep Research Bench 排名第6（RACE 0.4943），对标商业 Deep Research 产品

### 整体架构

Supervisor + Worker 模式（Lead Researcher 委派给多个 Sub-Researcher）：

```
User Input (含澄清问题或直接研究话题)
       │
  ┌────▼──────────────────────────────────────┐
  │  clarify_with_user (可选)                   │
  │  → {need_clarification, question}          │
  └────┬──────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────────┐
  │  transform_to_research_topic               │
  │  → 将对话转化为精确研究问题（第一人称，避免假设）│
  └────┬──────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────────┐
  │  Lead Researcher（Supervisor）              │
  │  - 使用 think_tool 规划研究策略              │
  │  - 调用 ConductResearch 委派子 agent        │
  │  - 最多 {max_researcher_iterations} 次      │
  │  - 最多 {max_concurrent_research_units} 并行 │
  └────┬──────────────────────────────────────┘
       │  并行分发
  ┌────▼──────────────────────────────────────┐
  │  Research Sub-Agents（每个独立子话题一个）    │
  │  - 使用 tavily_search（2-5次）              │
  │  - 使用 think_tool 反思搜索结果              │
  │  - 返回 compressed findings                │
  └────┬──────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────────┐
  │  compress_research                         │
  │  → 清理、去重、内联引用                       │
  └────┬──────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────────┐
  │  final_report_generation                   │
  │  → Markdown 格式，[Title](URL) 引用，与用    │
  │    户语言匹配                               │
  └───────────────────────────────────────────┘
```

**框架**：LangGraph Platform（可部署）  
**协作模式**：Supervisor（Lead Researcher）→ Parallel Sub-Agents → Sequential（压缩+报告生成）

### Agent 角色划分

| Agent | 职责 |
|-------|------|
| Lead Researcher（Supervisor） | 规划研究策略，委派子任务，决定何时完成 |
| Research Sub-Agent（多个） | 独立执行搜索任务（2-5次 Tavily），提取关键信息 |
| Compressor | 清理研究结果，保留原文，去重，添加内联引用 |
| Report Generator | 生成最终 Markdown 报告，匹配用户语言 |
| Webpage Summarizer | 将网页摘要到25-30%长度，提取关键引用 |

### 关键 System Prompt（原文引用）

**Lead Researcher Prompt（核心段落）：**
```
"You are a research supervisor. Your job is to conduct research by calling the 
'ConductResearch' tool. For context, today's date is {date}.

<Task>
Your focus is to call the 'ConductResearch' tool to conduct research against the overall 
research question passed in by the user. When you are completely satisfied with the research 
findings returned from the tool calls, then you should call the 'ResearchComplete' tool to 
indicate that you are done with your research.
</Task>

<Hard Limits>
- Bias towards single agent - Use single agent for simplicity unless the user request has 
  clear opportunity for parallelization
- Limit tool calls - Always stop after {max_researcher_iterations} tool calls to 
  ConductResearch and think_tool
- Maximum {max_concurrent_research_units} parallel agents per iteration
</Hard Limits>

<Show Your Thinking>
Before you call ConductResearch tool call, use think_tool to plan your approach:
- Can the task be broken down into smaller sub-tasks?
After each ConductResearch tool call, use think_tool to analyze the results:
- What key information did I find? What's missing? Do I have enough?
</Show Your Thinking>

<Scaling Rules>
Simple fact-finding: 1 sub-agent
Comparisons in user request: 1 sub-agent per element
When calling ConductResearch, provide complete standalone instructions - sub-agents 
can't see other agents' work
Do NOT use acronyms or abbreviations in your research questions, be very clear and specific
</Scaling Rules>"
```

**Research Sub-Agent Prompt（核心段落）：**
```
"You are a research assistant conducting research on the user's input topic.

<Hard Limits>
- Simple queries: Use 2-3 search tool calls maximum
- Complex queries: Use up to 5 search tool calls maximum
- Stop Immediately When: You can answer the user's question comprehensively; 
  You have 3+ relevant examples/sources; Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze:
- What key information did I find? What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>"
```

**Final Report Generation Prompt：**
```
"Generates comprehensive answers...requires markdown formatting, proper citations using 
[Title](URL) format, sources section, and critical emphasis: 
'Make sure the answer is written in the same language as the human messages!'"
```

### Tools 列表
- Tavily Search API（默认）
- Anthropic native search（Claude 原生搜索）
- OpenAI search
- MCP 服务器（任意 MCP 兼容工具）
- think_tool（内部反思工具）

### 使用模型
- 最高性能：GPT-5（RACE 0.4943）
- 可配置：GPT-4.1、Claude Sonnet 4
- 支持：OpenRouter、Ollama

### 输出格式
- 多层级 Markdown 报告
- 内联引用 `[Title](URL)` 格式
- 完整 Sources 章节
- 语言与用户输入语言匹配

### 与 gtm-team 对比亮点
- **think_tool（Chain-of-Thought 工具化）**：将 CoT 推理显式工具化，让 LLM 在搜索前后都进行结构化思考，大幅提升研究质量。gtm-team 可以借鉴这种"反思节点"设计。
- **预算控制**：严格的搜索次数上限（简单 2-3 次，复杂 5 次）防止 agent 无限循环。
- **澄清 → 精确话题转化**：两步预处理，先澄清模糊需求，再将对话转化为精确研究问题，减少歧义。

---

## 5. guy-hartstein/company-research-agent

**链接**：https://github.com/guy-hartstein/company-research-agent  
**Stars**：1,700  
**最后更新**：2025-11-18（v2.0.0）  
**语言**：Python 60%, TypeScript 33.8%  
**Forks**：257  
**定位**：面向公司尽职调查的 LangGraph 多 agent 框架，最接近 GTM research 的工程实践参考

### 整体架构

Fan-out / Fan-in 模式：4个并行研究员 → 汇总 → 精选 → 富化 → 摘要 → 编辑

```
Input: 公司名称 + URL + 所在地 + 行业
       │
  ┌────▼─────────────────────────────────────────┐
  │  GroundingNode（初始化上下文）                   │
  └────┬─────────────────────────────────────────┘
       │  fan-out（并行）
  ┌────┴──────────────────────────────────────────┐
  │  ┌──────────────┐  ┌──────────────────────┐   │
  │  │ CompanyAnal- │  │  IndustryAnalyzer    │   │
  │  │ yzer         │  │  (市场定位+竞争格局)   │   │
  │  └──────────────┘  └──────────────────────┘   │
  │  ┌──────────────┐  ┌──────────────────────┐   │
  │  │FinancialAna- │  │   NewsScanner        │   │
  │  │ lyst         │  │  (最新公告+合作+荣誉)  │   │
  │  └──────────────┘  └──────────────────────┘   │
  └────┬──────────────────────────────────────────┘
       │  fan-in
  ┌────▼──────────────────────────────────────────┐
  │  Collector（汇总所有研究结果）                    │
  └────┬──────────────────────────────────────────┘
  ┌────▼──────────────────────────────────────────┐
  │  Curator（相关性过滤，minimum 0.4 阈值）          │
  └────┬──────────────────────────────────────────┘
  ┌────▼──────────────────────────────────────────┐
  │  Enricher（补充信息富化）                         │
  └────┬──────────────────────────────────────────┘
  ┌────▼──────────────────────────────────────────┐
  │  Briefing（生成分章节摘要）                        │
  └────┬──────────────────────────────────────────┘
  ┌────▼──────────────────────────────────────────┐
  │  Editor（最终报告格式化）                          │
  └────┬──────────────────────────────────────────┘
       │
  Output: 结构化公司报告 (+ 可选 PDF 导出)
```

**框架**：LangGraph + FastAPI + React + MongoDB（可选持久化）  
**协作模式**：Parallel（4 Researchers）→ Sequential（Collector → Curator → Enricher → Briefing → Editor）

### Agent 角色划分

| Agent / Node | 职责 | System Prompt 关键内容 |
|---|---|---|
| GroundingNode | 初始化调查上下文 | "Expert researcher starting investigation" |
| CompanyAnalyzer | 公司基本信息研究 | 生成 queries：核心产品、公司历史、Leadership、商业模式 |
| IndustryAnalyzer | 行业分析 | 市场定位、竞争对手、行业趋势、市场规模 |
| FinancialAnalyst | 融资历史 | 融资轮次（含日期和投资方名称）、营收模式 |
| NewsScanner | 最新新闻 | 重大公告、合作、获奖，按时间倒序排列 |
| Collector | 汇总 | — |
| Curator | 相关性评分过滤（阈值 0.4） | — |
| Enricher | 补充数据 | — |
| Briefing | 分章节摘要生成 | — |
| Editor | 最终报告编辑 | "You are an expert report editor that compiles research briefings into comprehensive company reports." |

### 关键 System Prompt（原文引用）

**Editor System Message：**
```
"You are an expert report editor that compiles research briefings into 
comprehensive company reports."
```

**Content Sweep System Message（格式统一）：**
```
"You are an expert markdown formatter that ensures consistent document structure."
```

**Company Briefing Prompt（结构化输出）：**
要求输出章节：Core Product/Service、Leadership Team、Target Market、Key Differentiators、Business Model（全部 bullet-point，无解释性文字）

**Financial Briefing：**
融资轮次（含日期和投资方名称）+ 营收模式

**News Briefing：**
按时间倒序排列的：Major Announcements、Partnerships、Recognition

### Tools 列表
- **Tavily API**（相关性评分搜索，最低阈值 0.4）
- **FastAPI**（后端 REST API）
- **React**（前端 UI）
- **MongoDB**（可选持久化）
- **PDF 生成**（`/generate-pdf` 端点）

### 使用模型
- Google Gemini 2.5 Flash（大量文本合成）
- OpenAI GPT-5.1（格式化和编辑）

### 输出格式
- Markdown 结构化报告
  - Company Overview
  - Industry Overview
  - Financial Overview
  - News
  - References
- 禁止代码块和元评论
- 可导出 PDF

### 与 gtm-team 对比亮点
- **双模型策略**：Gemini 2.5 Flash 处理大量内容合成，GPT-5.1 负责最终格式化——成本与质量的平衡。
- **Curator 节点**：基于相关性评分（0.4 阈值）自动过滤噪音数据，gtm-team 可借鉴防止 hallucination。
- **完整的 REST API + React 前端**：工程化程度最高的开源 GTM research 项目之一。

---

## 6. ferdinandobons/startup-skill

**链接**：https://github.com/ferdinandobons/startup-skill  
**Stars**：261  
**最后更新**：2026-03-24（v1.7.0）  
**语言**：Markdown + Python  
**License**：MIT  
**定位**：专门为 Claude Code 等编码 agent 设计的 GTM/竞品分析技能包（Skill 模式）

### 整体架构

Skill 文件驱动架构（非传统代码框架）：每个 Skill 是一个结构化 Markdown 文件，由 Agent 按步骤执行：

```
用户调用 /startup:startup-competitors
       │
  ┌────▼──────────────────────────────────────────────────┐
  │  Phase 1: Intake（问答或读取已有文件）                      │
  │  - 检测 01-discovery/、00-intake/ 中是否有已有分析文件      │
  │  - 若无，执行 Round 1/Round 2 问答                        │
  └────┬──────────────────────────────────────────────────┘
  ┌────▼──────────────────────────────────────────────────┐
  │  Phase 1.5: Research Depth Assessment（1-9分）          │
  │  - 市场广度 + 已知竞品数量 + 地域范围                       │
  │  - Light(3-4) / Standard(5-7) / Deep(8-9)             │
  └────┬──────────────────────────────────────────────────┘
  ┌────▼──────────────────────────────────────────────────┐
  │  Phase 2: Three Parallel Research Waves               │
  │  ┌─────────────────────────────────────────────────┐  │
  │  │ Wave 1:                                          │  │
  │  │   A1 - Competitor Deep-Dives (5-8 direct + 2-3) │  │
  │  │   A2 - Pricing Intelligence (逆向工程定价体系)     │  │
  │  ├─────────────────────────────────────────────────┤  │
  │  │ Wave 2:                                          │  │
  │  │   B1 - Review Mining (G2/Capterra/TrustRadius)   │  │
  │  │   B2 - Forum & Community Mining (Reddit/IH/HN)   │  │
  │  ├─────────────────────────────────────────────────┤  │
  │  │ Wave 3:                                          │  │
  │  │   C1 - GTM Analysis (渠道+销售动作)               │  │
  │  │   C2 - Strategic & Growth Signals (融资+招聘)     │  │
  │  └─────────────────────────────────────────────────┘  │
  └────┬──────────────────────────────────────────────────┘
  ┌────▼──────────────────────────────────────────────────┐
  │  Phase 3: Synthesis（综合6个 Wave 的输出）                │
  │  - 连接跨 Wave 发现（定价缺口 + 用户抱怨 + 招聘 = 机会）     │
  │  - 置信度标注：[Data] [Estimate] [Assumption] [Opinion] │
  └────┬──────────────────────────────────────────────────┘
  ┌────▼──────────────────────────────────────────────────┐
  │  Phase 3.5: Verification Agent（独立核查）               │
  │  - 检查无标注声明、矛盾、数据过期、重复引用等               │
  └────────────────────────────────────────────────────── ┘
```

**框架**：Claude Code / Cursor 等 Skill 系统  
**协作模式**：Sequential（Phases）+ Parallel（Wave 内 Agent 对）

### Agent 角色划分

| Wave Agent | 职责 | 数据源 |
|---|---|---|
| A1 - Competitor Deep-Diver | 5-8 直接竞品画像（功能、团队、融资、牵引信号） | 官网、招聘页、融资数据、评论 |
| A2 - Pricing Intelligence | 逆向工程定价模型（价值度量、分层差异、定价心理） | 定价页、对话记录 |
| B1 - Review Miner | 挖掘用户好评/差评/功能请求 | G2、Capterra、TrustRadius、Product Hunt、App Store |
| B2 - Forum Miner | 抓取用户抱怨、迁移故事、变通方案 | Reddit、Indie Hackers、HN、Quora |
| C1 - GTM Analyst | 主要获客渠道、销售动作（自助 vs 销售驱动）、内容策略 | 社媒、SEO、广告投放 |
| C2 - Strategic Signals | 融资轨迹、招聘模式、产品路线图信号 | Crunchbase、LinkedIn Jobs、Changelog |
| Verification Agent | 独立核查事实准确性 | 对比所有 Wave 结果 |

### 关键 Prompt / 设计原则（原文引用）

**Honesty Protocol（诚实协议，贯穿全流程）：**
```
1. No cheerleading — acknowledge when competitors objectively excel
2. Label claims: [Data], [Estimate], [Assumption], [Opinion]
3. Quantify specifics — e.g., "$12M ARR, 40% YoY growth" not "growing fast"
4. Date everything — flag data older than 12 months
5. Declare gaps — "DATA GAP: Could not find reliable data on [X]" beats fabrication
6. Surface red flags — state if competitive landscape is brutal
7. Challenge confirmation bias — probe deeper when research confirms founder beliefs
```

**Research Depth Scoring：**
```
Score = Market Breadth(1-3) + Known Competitors(1-3) + Geographic Scope(1-3)
Light(3-4) / Standard(5-7) / Deep(8-9)
```

### Tools 列表
- Web 搜索（通过 Claude Code 的 Agent 工具）
- 评论平台（G2、Capterra、TrustRadius、Product Hunt、App Store）
- 社区平台（Reddit、Indie Hackers、Hacker News、Quora）
- 融资/招聘数据（Crunchbase、LinkedIn）

### 使用模型
Claude Code（Claude Max 5x 推荐，因 token 消耗大）

### 输出格式
```
{project-name}/
├── intake.md
├── PROGRESS.md
├── competitors-report.md       ← 执行摘要 + 战略机会/风险
├── competitive-matrix.md       ← 功能矩阵（强/中/弱/缺失）
├── pricing-landscape.md        ← 定价层级对比 + 定价空白
├── battle-cards/{competitor}.md ← 每个竞品一张 battle card
└── raw/
    ├── competitor-profiles.md
    ├── pricing-intelligence.md
    ├── review-mining.md
    ├── forum-mining.md
    ├── gtm-analysis.md
    └── strategic-signals.md
```

### 与 gtm-team 对比亮点
- **6-Wave 并行竞品研究**是迄今最系统的竞品分析框架：覆盖产品、定价、用户情绪、社区声音、GTM 策略、战略信号六个维度。
- **置信度标注系统**（[Data]/[Estimate]/[Assumption]/[Opinion]）是防止 hallucination 的工程最佳实践。
- **Battle Card 输出**：直接生成可用于销售的竞品 battle card，是 GTM research 的关键交付物之一。

---

## 7. MaxKmet/idea-validation-agents

**链接**：https://github.com/MaxKmet/idea-validation-agents  
**Stars**：93  
**最后更新**：2026-04-15  
**语言**：Markdown（Skill 文件）  
**License**：MIT  
**定位**：面向独立开发者的创业想法验证 + GTM 策略 agent，Claude Code / Cursor 驱动

### 整体架构

四大工作流（Intent Router 路由）：

```
用户 Query
   │
   ▼ Intent Router (CLAUDE.md 定义)
   ├── 没有想法 → Idea Generation Workflow (~10-15min)
   │              面试用户 → 分析趋势 → 生成 7-10 个打分想法
   ├── 有想法 → Idea Validation Workflow (~10-15min)
   │              9步验证：TAM/SAM/SOM + 竞品 + RAT 实验设计
   ├── 想法失败 → Pivot Optimization Workflow (~10-15min)
   │              1-2 个变量调整（受众/细分/定价/功能）+ 预测分数提升
   └── 市场研究 → Market Deep-Dive Workflow (~10-15min)
                  类目研究 + 竞品格局 + 渠道评估
```

**协作模式**：Sequential（按 Workflow 步骤），记忆持久化到 `memory/` 文件夹

### Agent 角色划分
系统是单一编排 agent（Claude Code），通过不同 Skill 文件切换执行模式：
- **Trend Analyzer**：分析 TikTok Creative Center、Reddit、App Store、Google Trends
- **Competitor Mapper**：竞品格局与 TAM/SAM/SOM 建模
- **Pricing Analyst**：Van Westendorp 定价分析
- **Distribution Analyst**：渠道评估（ASO、病毒循环建模）
- **Risk Assessor**：Pre-mortem 分析
- **Pivot Optimizer**：识别改进路径

### 关键 System Prompt（CLAUDE.md 核心内容引用）

```
核心设计原则（来自 CLAUDE.md）：
- 窄技能函数，结构化输出（JSON/Markdown）
- 数据驱动挑战而非安慰，锚定竞争证据和类目基准
- 真实市场信号优于猜测
- 使用乘法下限算法：一个灾难性弱点会大幅压低最终分数
- RAT（Riskiest Assumption Test）：≤2周，≤$100 的验证实验设计

面试结构：
- Full（10问）/ Fast（4问）/ Browse（话题选择）/ Skip（最少）模式
- 技术能力评估在所有模式中必须执行

内存架构：
- 用户画像层（user_profile.md）
- 市场洞察层（按细分/平台/时期）
- 每个想法状态目录（competitors.json, pricing.json, scores.json, decision_memo.md）
```

### Tools 列表
- TikTok Creative Center（趋势数据）
- Reddit 搜索（用户需求信号）
- App Store（竞品评分+评论）
- Google Trends（搜索趋势）
- Van Westendorp 定价模型
- 病毒循环建模（6种循环类型）

### 使用模型
Claude Code（Claude Max 推荐）、OpenAI Codex、Cursor

### 输出格式
```
memory/
├── user_profile.md
├── market_insights/
└── ideas/{category}/
    ├── competitors.json
    ├── pricing.json
    ├── scores.json
    └── decision_memo.md
```

### 与 gtm-team 对比亮点
- **乘法下限评分算法**：一个关键弱点（如市场过饱和）会大幅压低总分，比线性加权更真实。
- **RAT（最危险假设测试）**：每个想法自动设计一个≤$100的验证实验，将分析转化为行动。
- **持久化内存架构**：跨 session 的记忆系统，适合长期迭代的 GTM 规划场景。

---

## 8. chadboyda/agent-gtm-skills

**链接**：https://github.com/chadboyda/agent-gtm-skills  
**Stars**：36  
**最后更新**：2026年初  
**语言**：Markdown（Skill 文件）  
**License**：MIT  
**定位**：18个独立 GTM 技能包，覆盖完整收入运营层面，每个 Skill 300-900 行

### 整体架构

独立 Skill 文件库（无 Agent 框架依赖）：

```
skills/
├── strategy/
│   ├── positioning-icp/SKILL.md     ← ICP定义 + 定位 + PMF验证
│   ├── ai-pricing/SKILL.md          ← 定价策略
│   └── sales-motion-design/SKILL.md ← 销售动作设计
├── outbound/
│   ├── ai-cold-outreach/SKILL.md    ← 6段漏斗式冷外联
│   ├── ai-sdr/SKILL.md              ← AI SDR 自动化
│   ├── lead-enrichment/SKILL.md     ← 数据富化瀑布流
│   └── video-outreach/SKILL.md      ← 个性化视频外联
├── inbound/
│   ├── multi-platform-launch/SKILL.md
│   ├── ai-seo/SKILL.md
│   ├── social-selling/SKILL.md
│   └── content-to-pipeline/SKILL.md
├── paid/
│   ├── ai-ugc-ads/SKILL.md
│   └── paid-creative-ai/SKILL.md
├── retention/
│   ├── expansion-retention/SKILL.md
│   └── partner-affiliate/SKILL.md
└── operations/
    ├── gtm-engineering/SKILL.md
    ├── solo-founder-gtm/SKILL.md
    └── gtm-metrics/SKILL.md
```

**协作模式**：独立 Skill 调用，Skill 内定义了集成流程

### 关键 Prompt 设计（positioning-icp SKILL.md 核心片段）

```
Four-Layer Positioning Stack:
- Category: Market context buyers understand
- Wedge: Specific capability gap exploited
- Proof Vector: Quantified evidence of results
- Alternative Framing: Competitive positioning for search

Three-Signal ICP Definition:
- Firmographic signals (company shape, industry, size)
- Technographic signals (tech stack, API readiness)
- Intent signals (active buying behavior, triggers)

Fit Score = (Firmographic 40%) + (Technographic 35%) + (Behavioral 25%)
Intent Score = (First-Party 40%) + (Third-Party 35%) + (Triggers 25%)

Enrichment Waterfall（停在置信度 0.85+）:
Clay → Apollo → ZoomInfo → BetterContact
- 0.85-1.00: Route to outreach
- 0.70-0.84: Accept with verification flag
- 0.50-0.69: Nurture only, no cold email
- Below 0.50: Reject
```

**Cold Outreach SKILL.md（6段漏斗）：**
```
Stage 1: Signal Detection（融资公告、招聘信号、意图数据）
Stage 2: Waterfall Enrichment（3-5个数据提供商，首验即停）
Stage 3: AI Personalization（$0.01-0.03/lead，基于公司研究生成个性化开场）
Stage 4: Sequencing（4-7封邮件，14-25天，条件逻辑分支）
Stage 5: Infrastructure（域名轮换、邮箱管理、认证协议）
Stage 6: Follow-up（AI回复分类：正向/疑问/异议/转介/拒绝）

First-line prompt:
"Research this company using provided data. Write one 1-sentence observation about 
[specific context]. Avoid corporate jargon."

Timeline hook approach（最高转化率）:
"When teams your size approach [milestone], most spend [time] on [problem]."
```

### Tools 列表
| 类别 | 工具 |
|------|------|
| 数据富化 | Clay（150+数据源）、Apollo、ZoomInfo、BetterContact |
| 邮件发送 | Instantly、Smartlead |
| 意图信号 | Bombora、G2、BuiltWith |
| AI 个性化 | Claude Sonnet 3.7（默认）、GPT-4o |
| 邮件验证 | ZeroBounce、NeverBounce |
| 视频外联 | Tavus、Sendspark、HeyGen |
| 合作伙伴 | PartnerStack、Impact |
| SEO | DataForSEO + Claude Code |
| 自动化 | n8n、Make、Zapier |

### 使用模型
Claude Sonnet（Claude Code）、GPT-4o（备选）

### 输出格式
每个 Skill 是独立的 Markdown 分析报告或操作手册

### 与 gtm-team 对比亮点
- **最完整的 GTM 覆盖**：唯一覆盖"战略 → 外联 → 内容 → 付费 → 留存 → 运营"全链路的开源 GTM 技能集。
- **数据富化瀑布流设计**：按置信度分级路由（0.85+直接外联，0.50-以下拒绝），工程化防止资源浪费。
- **现代 2025-2026 基准数据**：含实际 KPI 基准（回复率目标 5-10%，每次会议成本 $3-36 等），可直接用于 GTM 策略校准。

---

## 9. psrane8/Market-Research-Agent

**链接**：https://github.com/psrane8/Market-Research-Agent  
**Stars**：22  
**语言**：Python  
**定位**：CrewAI 多 agent 框架的公司市场+财务分析

### 整体架构

Sequential 3-Agent 流水线（CrewAI）：

```
Input: 公司名称 {company}
       │
  ┌────▼──────────────────────────┐
  │  Market Research Analyst      │ ← Task 1（异步）
  │  "Provide insights about      │
  │  {company} through market     │
  │  analysis"                    │
  └────┬──────────────────────────┘
       │ + 并行
  ┌────▼──────────────────────────┐
  │  Financial Analyst            │ ← Task 2（异步）
  │  "Provide comprehensive       │
  │  financial insights about     │
  │  {company}"                   │
  └────┬──────────────────────────┘
       │ (等待两者完成)
  ┌────▼──────────────────────────┐
  │  Reporting Analyst            │ ← Task 3（同步，依赖前两个）
  │  "Create sophisticated        │
  │  reports based on the         │
  │  findings..."                 │
  └────┬──────────────────────────┘
       │
  Output: 2页综合报告（市场+财务）
```

**框架**：CrewAI + LangChain  
**协作模式**：Parallel（Task 1+2）→ Sequential（Task 3）

### Agent 角色划分

| Agent | Goal（CrewAI Backstory） | Memory |
|-------|--------------------------|--------|
| Market Research Analyst | "Provide insights about {company} through market analysis"；分析市场趋势、消费者行为、竞争动态 | 启用 |
| Financial Analyst | "Provide comprehensive financial insights about {company}"；评估季度业绩、编制财务预测 | 启用 |
| Reporting Analyst | "Create sophisticated reports based on the findings from financial and market research analysts about {company}"；不可委托（no delegation） | 启用 |

### 关键任务定义（原文引用）

**Task 1 - Financial Analysis：**
```
description: "Analyze the financial performance of {company}"
expected_output: "A detailed financial report including key financial ratios, trends, and forecasts"
```

**Task 2 - Market Analysis：**
```
description: "Analyze market trends and competitive landscape for {company}"
expected_output: "A comprehensive market research report detailing market trends, 
consumer behavior, and competitive analysis"
```

**Task 3 - Reporting：**
```
description: "Compile and synthesize data from financial and market research analysts 
into a comprehensive report"
expected_output: "A detailed 2 page report that combines financial performance analysis 
with market trends and competitive analysis for {company}, 
all important findings should be highlighted"
```

### Tools 列表
- SerperAPI（Google 搜索）
- 所有 agent 共用同一个 `tool`（Serper）

### 使用模型
Google Gemini 1.5 Flash（temperature=0.5）

### 输出格式
2页综合报告（Markdown）

---

## 10. SalmaSalahEldin/Multi-Agent-Market-Research

**链接**：https://github.com/SalmaSalahEldin/Multi-Agent-System-for-AI-Powered-Market-Research-Using-LangGraph-and-LangChain  
**Stars**：7  
**语言**：Python  
**定位**：LangGraph + ReAct 架构的市场研究 + AI 用例生成 + 资源收集三阶段流水线

### 整体架构

Sequential 3-Agent 链（LangGraph ReAct）：

```
Input: 行业/公司名称
       │
  ┌────▼──────────────────────┐
  │  Research Agent (ReAct)   │
  │  system: "market research │
  │  expert..."               │
  │  tool: TavilySearch(k=3)  │
  └────┬──────────────────────┘
       │ (content as HumanMessage)
  ┌────▼──────────────────────┐
  │  Use Case Agent (ReAct)   │
  │  system: "AI use case     │
  │  generation expert..."    │
  │  tool: TavilySearch(k=3)  │
  └────┬──────────────────────┘
       │ (content as HumanMessage)
  ┌────▼──────────────────────┐
  │  Resource Agent (ReAct)   │
  │  system: "resource asset  │
  │  collection expert..."    │
  │  tool: TavilySearch(k=3)  │
  └────┬──────────────────────┘
       │
  Output: 时间戳命名的 Markdown 文件
```

**框架**：LangGraph（每个 agent 独立 StateGraph）  
**协作模式**：Sequential（通过 HumanMessage 传递上下文）

### 关键 System Prompt（原文引用）

**Research Agent：**
```
"You are a market research expert specializing in industry analysis and competitive insights...
Understand the Industry and Segment, competitor analysis with AI/GenAI examples, 
quantitative metrics, and Actionable Insights highlighting automation opportunities."
```

**Use Case Agent：**
```
"You are an expert in AI use case generation, specializing in industry-specific innovation...
Focus exclusively on use cases that leverage AI, ML, GenAI, and automation; structured 
format with Objective, Application, and Cross-Functional Benefits; minimum five use cases required."
```

**Resource Agent：**
```
"You are an expert in resource asset collection, tasked with identifying datasets, tools, 
frameworks... Search platforms like Kaggle and HuggingFace; identify pre-trained models, 
APIs, or open-source tools; propose GenAI solutions including document search and automated 
report generation."
```

### Agent 内部 ReAct 图结构
```python
AgentState: messages: Annotated[list[AnyMessage], operator.add]

Nodes: "llm" → conditional → "action" → "llm" (loop)
Entry: "llm"
Tool: TavilySearchResults(max_results=3)
```

### Tools 列表
- TavilySearchResults（所有 agent 共用，max_results=3）

### 使用模型
GPT-4（OpenAI）

### 输出格式
时间戳命名的 Markdown 文件（含市场研究、用例列表、资源集合三部分）

---

## 11. ramamoorthy07/Multi-Agent-Market-Research-and-Use-Case

**链接**：https://github.com/ramamoorthy07/Multi-Agent-Market-Research-and-Use-Case-Generation-System  
**Stars**：5  
**语言**：Python  
**定位**：CrewAI 4-agent 市场研究 + AI 用例生成系统，Streamlit 界面

### 整体架构

Sequential 4-Agent CrewAI（按依赖顺序）：

```
Input: 目标公司 + 行业
       │
  ┌────▼─────────────────────────────┐
  │  industry_researcher              │
  │  Role: "Industry Research         │
  │  Specialist"                      │
  │  Tools: search_tool, scrape_tool  │
  └────┬─────────────────────────────┘
       │
  ┌────▼─────────────────────────────┐
  │  use_case_generator               │
  │  Role: "AI Use Case Strategist"   │
  │  Tools: search_tool, scrape_tool, │
  │         pdf_tool                  │
  └────┬─────────────────────────────┘
       │
  ┌────▼─────────────────────────────┐
  │  resource_collector               │
  │  Role: "AI Resource Specialist"   │
  │  Tools: search_tool, scrape_tool  │
  └────┬─────────────────────────────┘
       │
  Output: Streamlit 展示的结构化报告
```

**框架**：CrewAI + Google Generative AI + Streamlit  
**协作模式**：Sequential（无委托，allow_delegation=False）

### 关键 Agent 定义（原文引用）

**Industry Researcher：**
```python
role="Industry Research Specialist"
goal="Analyze the specified company's sector to identify market trends, competitor 
strategies, positioning opportunities, key offerings, recent advancements, and growth areas."
backstory="An experienced industry analyst specializing in competitive landscape 
assessment and strategic opportunity identification for decision-makers."
tools=[search_tool, scrape_tool]
```

**Use Case Strategist：**
```python
role="AI Use Case Strategist"
goal="Research emerging industry trends...identifying AI/ML applications that provide 
competitive advantages and generating innovative use cases emphasizing efficiency, 
scalability, and ROI."
backstory="Expert in GenAI, AI, and ML who translates technological advancements into 
practical business solutions, bridging innovation and real-world application."
tools=[search_tool, scrape_tool, pdf_tool]
```

**Resource Collector：**
```python
role="AI Resource Specialist"
goal="Compile curated datasets, libraries, and AI/ML tools essential for implementing 
proposed use cases, ensuring relevance to industry and practical integration capability."
backstory="AI resource expert sourcing tools, datasets, and libraries aligned with 
strategic project goals."
tools=[search_tool, scrape_tool]
```

### Tools 列表
- SerperDevTool（Google 搜索）
- ScrapeWebsiteTool（网站爬取）
- PDFSearchTool（PDF 分析）

### 使用模型
Google Gemini Flash 1.5

---

## 12. Nirikshan95/VettIQ

**链接**：https://github.com/Nirikshan95/VettIQ  
**Stars**：13  
**语言**：Python  
**框架**：LangGraph + FastAPI + Streamlit  
**定位**：AI 创业想法验证平台，包含完整的 LangGraph 工作流 + 结构化 System Prompt

### 整体架构

LangGraph Sequential Workflow（含 Tools 节点和 Fallback）：

```
Input: startup_idea
       │
  ┌────▼──────────────────────┐
  │  Market Analyst           │
  │  → tools node (if needed) │
  │  → fallback to chat       │
  └────┬──────────────────────┘
       │ market_analysis
  ┌────▼──────────────────────┐
  │  Competitor Analyst       │
  │  → tools node             │
  └────┬──────────────────────┘
       │ competition_analysis
  ┌────▼──────────────────────┐
  │  Risk Assessor            │
  │  → tools node             │
  └────┬──────────────────────┘
       │ risk_assessment
  ┌────▼──────────────────────┐
  │  Strategic Advisor        │
  │  → Go/No-Go 决策           │
  └────┬──────────────────────┘
       │
  Output: 结构化建议 + Streamlit 展示
```

### 关键 System Prompt（原文引用）

**Market Analyst：**
```
"You are a Market Analyst for startups. Analyze the target market for the given startup 
idea: {startup_idea}. Identify customer segments, estimate market size, highlight key 
trends, and note opportunities or gaps.

IMPORTANT:
- If you decide to use tools, make tool_calls only and completely ignore generating text 
  in the content field.
- If you decide to answer directly (without tool calls), respond in a clear, well-structured 
  plain text format. Use bullet points for clarity. Keep the response concise, realistic, 
  and directly actionable."
```

**Competitor Analyst：**
```
"You are a Competitive Intelligence Analyst.
Your task is to analyze the competitive landscape for the following startup idea 
and its market context.

Startup Idea: {startup_idea}
Market Analysis: {market_analysis}

Please identify:
- Direct and indirect competitors
- Their main features and offerings
- Pricing details if available
- Strengths and weaknesses of key competitors
- Key differentiators of the proposed startup
- Strategic positioning recommendations

Only include realistic and relevant insights. The final response must be written as 
plain text suitable for storage in a single string field named competition_analysis."
```

**Risk Assessor：**
```
"You are a Risk Assessor for startups. Evaluate the following: Startup Idea, Market 
Analysis, Competitor Analysis. Identify key risks in Market, Technical, Operational, 
Regulatory, and Financial categories. For each category, include: description, severity 
(Low/Medium/High), and mitigation strategy."
```

**Strategic Advisor：**
```
"You are a Startup Advisor.
Review the idea, market analysis, competitor analysis, and risk assessment.
Give a clear Go/No-Go (or Conditional Go) recommendation with reasoning.

Recommendation:
- Decision: "Go" / "No-Go" / "Conditional Go"

advice:
- Suggested improvements or next steps if your Recommended Decision is "Go" 
  or "Conditional Go"
- else critique or reasons for why it is not good to "Go"

format instructions are: {format_instructions}
```

### Tools 列表
- DuckDuckGo Search（`duckduckgo-search` 库）
- Tools-first with fallbacks（工具失败自动降级为 chat-only）

### 使用模型
Hugging Face endpoint（默认 `openai/gpt-oss-120b`）

---

## 13. CrewAI 官方示例 - Marketing Strategy Crew

**链接**：https://github.com/crewAIInc/crewAI-examples/tree/main/crews/marketing_strategy  
**Stars**：（CrewAI Examples 仓库整体）  
**语言**：Python  
**定位**：CrewAI 官方营销策略多 agent 示例

### 整体架构

Sequential 4-Agent CrewAI（含 Supervisor 风格的 Chief Creative Director）：

```
Input: {customer_domain} + {project_description}
       │
  ┌────▼────────────────────────────────────────────────┐
  │  Lead Market Analyst                                 │
  │  Task: research_task                                 │
  │  "Investigate customer domain and competitors within │
  │  2024 context"                                       │
  └────┬────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │  Lead Market Analyst                                 │
  │  Task: project_understanding_task                    │
  │  "Analyze project specifics and target audience"     │
  └────┬────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │  Chief Marketing Strategist                          │
  │  Task: marketing_strategy_task                       │
  │  输出: goals + target audience + key messages +      │
  │  proposed tactics + name + channels + KPIs           │
  └────┬────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │  Creative Content Creator                            │
  │  Task: campaign_idea_task → copy_creation_task       │
  │  5个创意方案 + 文案                                   │
  └────┬────────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │  Chief Creative Director（QA 节点）                    │
  │  监督全团队输出质量与战略对齐                              │
  └────────────────────────────────────────────────────┘
```

### 关键 Agent 定义（agents.yaml，原文引用）

```yaml
lead_market_analyst:
  role: "Lead Market Analyst"
  goal: "Analyze products and competitors to deliver insights for marketing strategies"
  backstory: "Specializes in examining online business landscapes at a digital marketing firm"

chief_marketing_strategist:
  role: "Chief Marketing Strategist"
  goal: "Transform product analysis findings into effective marketing strategies"
  backstory: "Develops customized approaches known for driving organizational success"

creative_content_creator:
  role: "Creative Content Creator"
  goal: "Develop compelling and innovative content for social media campaigns"
  backstory: "Converts strategies into engaging narratives and visual content that 
  motivate audience action"

chief_creative_director:
  role: "Chief Creative Director"
  goal: "Oversee team output for quality assurance and strategic alignment"
  backstory: "Leads content creation at a branding agency, ensuring optimal output for clients"
```

### Tools 列表
- SerperDevTool（Google 搜索）
- ScrapeWebsiteTool（网站爬取）
- 默认 GPT-4o

---

## 14. 与当前 gtm-team 项目的综合对比

基于对 gtm-team 项目（LangGraph + asyncio，port 8091，/team/ 路由）的了解，以下是综合对比分析：

### 架构对比矩阵

| 维度 | gtm-team（当前） | 开源项目最佳实践 |
|------|----------------|----------------|
| **框架** | OpenClaw-native asyncio | LangGraph（open_deep_research、TradingAgents、company-research-agent） |
| **Agent 数量** | 未知（需确认） | 3-10 个专职 agent（最常见 3-5 个） |
| **Supervisor 模式** | 未知 | Lead Researcher（open_deep_research）、Chief Editor（GPT Researcher）、Chief Creative Director（CrewAI） |
| **并行执行** | 未知 | Fan-out（company-research-agent）、Wave 并行（startup-skill）、3 parallel analysts（TradingAgents） |
| **RAG 集成** | 有 rag_mgr.py | VettIQ 的 FAISS、GPT Researcher 的本地文档分析 |
| **语义缓存** | 有 semantic_cache.py | 大多数项目无此功能（差异化优势） |
| **Review 循环** | 未知 | Reviewer→Revisor Loop（GPT Researcher）、Verification Agent（startup-skill） |
| **输出格式** | 未知 | Battle Cards（startup-skill）、Fan-out/Fan-in 报告（company-research-agent）、多格式 PDF/DOCX/MD（GPT Researcher） |
| **人工反馈** | 未知 | Human-in-the-Loop（GPT Researcher）、VettIQ 的 Conditional Go |
| **模型配置** | 未知 | 双模型策略（Gemini Flash + GPT-5.1）、think_tool 反思 |

### gtm-team 可以借鉴的关键亮点

#### 1. Supervisor + Sub-Agent 模式（来自 open_deep_research）
最重要的架构建议：引入 Lead Researcher（Supervisor）负责研究规划和委派，Sub-Agents 并行执行具体搜索。当前 asyncio 架构可以直接实现这种模式。

关键设计：Sub-agents 接收完整的独立指令（cannot see other agents' work），防止上下文污染。

#### 2. think_tool 反思机制（来自 open_deep_research）
在每次搜索前后插入"思考节点"：
```
搜索前：Can this be broken into sub-tasks?
搜索后：What did I find? What's missing? Do I have enough?
```
这可以直接在 asyncio 流程中实现为 LLM 调用节点，显著提升研究逻辑性。

#### 3. Bull/Bear 辩论机制（来自 TradingAgents）
对 GTM 分析：引入"机会派 Agent"（强调市场机会、增长空间）和"风险派 Agent"（强调竞争、风险、挑战），由第三个 Agent 综合裁决，生成更平衡的分析。

#### 4. 6-Wave 竞品研究框架（来自 startup-skill）
直接可用的研究框架：
- Wave 1：竞品画像 + 定价逆向工程
- Wave 2：评论挖掘（G2/Capterra/Reddit/HN）
- Wave 3：GTM 策略 + 战略信号（融资/招聘）
+ Verification Agent 独立核查

#### 5. 置信度标注系统（来自 startup-skill）
所有 Agent 输出自动标注 `[Data]` / `[Estimate]` / `[Assumption]` / `[Opinion]`，防止 hallucination 混入最终报告。

#### 6. Battle Card 输出格式（来自 startup-skill、company-research-agent）
GTM research 的关键交付物：每个竞品一张 battle card（含：如何赢、何时输、客户异议处理、关键弱点、离网信号），直接服务销售团队。

#### 7. 双模型策略（来自 company-research-agent）
大量内容生成（Researcher 节点）用 Gemini Flash（快速+便宜），最终报告编辑（Editor 节点）用 GPT-4/5.1（高质量）。gtm-team 的 semantic_cache.py 可以在此基础上进一步优化成本。

#### 8. Enrichment Waterfall（来自 agent-gtm-skills）
对需要公司数据富化的场景：按置信度阈值依次使用 Clay → Apollo → ZoomInfo，首达 0.85 即停，避免重复验证。

### gtm-team 的差异化优势（保持并强化）
1. **semantic_cache.py**：大多数开源项目没有语义缓存，这是成本优化的重要机制。
2. **rag_mgr.py**：本地 RAG 集成（uploads/ 目录）支持私有文档分析，是开源项目普遍缺乏的。
3. **asyncio 原生架构**：比 LangGraph 更轻量，延迟更低，适合实时 API 场景。

---

*报告生成时间：2026-04-30*  
*数据来源：GitHub 官方页面、项目源代码文件（agents.py, prompts.py, SKILL.md, CLAUDE.md 等）、arXiv 论文*
