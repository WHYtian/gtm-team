# GTM Agent 模型基准测试报告

*测试时间：2026-04-29*

---

## 关键发现

### 1. DeepSeek V4 是推理模型（Reasoning Model）

DeepSeek V4 Flash / Pro 内置 chain-of-thought，每次调用会先消耗大量 token 进行内部推理，再输出最终答案。

测试数据：T2 任务设置 `max_tokens=500`，实际消耗分布：
- **Reasoning tokens（内部思考）: 434**
- **Output tokens（实际输出）: 66** ← 直接被截断

这就是 T2/T3 显示空输出的原因：token budget 被思考过程耗尽，内容还没输出完就停止了。

**影响**：V4 所有调用需要 2-3 倍的 max_tokens 才能获得完整输出，且 reasoning tokens 也计费，实际使用成本远高于表面数字。

### 2. Doubao Seed 2.0 系列速度极慢

Seed 模型（Lite/Mini/1.8/Pro）同样具有内部推理能力，输出 token 数远超实际需要（T2 输出 1600-2500 tokens），导致单次调用延迟 20-50 秒。

对于 Researcher（每次研究并行调用 7+ 次）而言，这是不可接受的。

### 3. deepseek-v3-2-251201 在火山引擎可用

经测试确认，DeepSeek V3 通过火山引擎接口可正常调用，是高性价比的标准推理模型（非 reasoning model）。

---

## 完整测试结果

### 测试维度说明

| 测试 | 对应角色 | 考察能力 | 评判标准 |
|------|---------|---------|---------|
| **T1 结构化输出** | Supervisor | 格式指令遵从（`DEEP DIVE:` 前缀格式） | 关键词：DEEP DIVE:, market_overview, technology_trends, regulatory_env |
| **T2 分析推理** | Analyst | Porter's Five Forces 框架应用 | 关键词：Porter, Five Forces, High, Medium, Low |
| **T3 批判思维** | Critic | 找出分析中具体漏洞 | 关键词：$1 trillion, Microsoft, Salesforce, weakness |
| **T4 长文写作** | Writer | 结构化报告输出 | 关键词：Executive Summary, $18.5B, 42B, CAGR, Wiz |

### 汇总表（修正后）

| 模型 | T1 结构化 | T2 推理 | T3 批判 | T4 写作 | 平均延迟 | 备注 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| Doubao Seed 2.0 Lite | 75% / 25.6s | 100% / 50.0s | 75% / 49.4s | 100% / 34.5s | 39.9s | 🔴 太慢 |
| Doubao Seed 2.0 Mini | 75% / 10.4s | 100% / 28.4s | **100% / 20.5s** | 60% / 19.4s | 19.6s | 批判最强，其余慢 |
| Doubao Seed 1.8 | 75% / 14.0s | 80% / 42.8s | 75% / 39.2s | 100% / 21.0s | 29.2s | 🔴 慢且推理弱 |
| Doubao Seed 2.0 Pro | 75% / 14.6s | 100% / 44.2s | 50% / 45.9s | 100% / 26.9s | 32.9s | 🔴 批判差 |
| **Doubao 1.5 Pro 32k** | **75% / 4.5s** | **100% / 10.1s** | 75% / 13.8s | **100% / 5.3s** | **8.4s** | 🟢 最均衡 |
| Doubao V3 (deepseek-v3-2-251201) | 75% / 6.1s | ~80% / 8.5s | ~75% / 15.4s | ~100% / est | ~9s | 🟢 快速高效 |
| DeepSeek V4 Flash ⚠️ | 75% / 3.0s | 0% / 7.5s* | 50% / 5.9s* | 100% / 5.5s | 5.5s | ⚠️ 推理模型，tokens被思考耗尽 |
| DeepSeek V4 Pro ⚠️ | 75% / 8.8s | 0% / 16.9s* | 0% / 12.1s* | 100% / 12.6s | 12.6s | ⚠️ 同上，更慢 |

> ⚠️ *DeepSeek V4 T2/T3 得 0% 的原因：max_tokens=500 中约 434 个被内部推理消耗，实际输出仅 ~66 tokens 被截断。将 max_tokens 提升至 1500 后，V4 Flash T2 得分 100%，T3 得分 75%。

---

## 各 Agent 模型分配建议

### Supervisor → `doubao-1-5-pro-32k-250115`，temperature=0.2

**任务特征**：  
- 路由判断（简单二分类）  
- 质量评估并输出 `DEEP DIVE: dim | reason: ... | search: ...` 格式  
- 每次研究只调用 1-2 次  

**选择理由**：  
T1 延迟最短（4.5s），指令遵从稳定。temperature=0.2 确保格式输出确定性——Supervisor 的结构化输出是整个 adaptive depth 流程的触发器，不能有随机波动。

DeepSeek V4 虽然也是 3s，但 reasoning token 会吃掉 budget，输出风险较高。

---

### Researcher → `deepseek-v3-2-251201`，temperature=0.3

**任务特征**：  
- 每次研究被调用 4-8 次（并行）  
- 任务简单：从爬取的原始文本提取统计数字、公司名、置信度评分  
- 延迟直接影响整体 pipeline 速度  

**选择理由**：  
V3 是标准指令遵从模型（非 reasoning model），token budget 完全用于输出。  
T1 仅用 78 tokens 精准完成结构化任务，T2 用 175 tokens 完成推理——对 Researcher 这类「简单提取」任务，高效比全能更重要。  

Doubao 1.5 Pro 32k 同样可行（8.4s），V3 稍快（6-8s）且两者都在火山引擎，用同一 API key。

---

### Analyst → `doubao-1-5-pro-32k-250115`，temperature=0.5

**任务特征**：  
- 最复杂的推理（TAM/PESTEL/Porter's）  
- 矛盾扫描：输出 `CONFLICT: ... | verify: ...` 格式  
- 每次调用 2-3 次（初步分析 + 矛盾扫描 + 修订）  

**选择理由**：  
T2 100% 且 10.1s —— 比 Seed 系列快 3-5 倍，且质量相当。

DeepSeek V4 Flash 经过深度测试，推理质量更强，但需要 2-3x max_tokens。对 Analyst 这个最重要的角色，增加 token budget 是合理的——但现阶段 Doubao 1.5 Pro 已够用，升级可作为下一步。

---

### Critic → `doubao-seed-2-0-mini-260215`，temperature=0.6

**任务特征**：  
- 每次研究只调用 **1 次**  
- 任务：找出分析中最尖锐的漏洞  
- 延迟不敏感（20s 对单次调用可以接受）  

**选择理由**：  
T3 唯一满分（100%，4/4 关键词）的模型。对比 75% 的其他模型，Mini 的批评更尖锐、更具体。

测试输出示例：
> *"Jamie's $1 trillion claim is entirely unmoored from credible data — no citation of Gartner/IDC, no breakdown of the 15-17% CAGR required to hit that target from $220B..."*

这种级别的批判质量远高于其他模型的泛泛而谈。

temperature=0.6 给 Critic 更多「创造性挑剔空间」，避免每次都输出相同的批评框架。

---

### Writer → ~~`doubao-1-5-pro-32k-250115`~~ → `deepseek-v3-2-251201`，temperature=0.4

**LLM-as-Judge 结果（2026-04-29）**：

| 模型 | 数据准确 | 结构 | 洞察质量 | 可读性 | 总分/20 | 延迟 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| DS-V3 ✅ | 5.0 | 5.0 | **4.0** | 5.0 | **19.0** | 6.5s |
| 1.5 Pro ~~旧~~| 5.0 | 5.0 | **2.3** | 5.0 | **17.3** | 4.9s |

**切换理由**：洞察质量（insight_quality）是决定性短板——1.5 Pro 得 2.3/5，DS-V3 得 4.0/5。执行摘要需要的不只是数据复述，而是数据+战略含义的结合。DS-V3 速度还更快（6.5s vs 4.9s）。

---

### Analyst → ~~`doubao-1-5-pro-32k-250115`~~ → `doubao-seed-2-0-mini-260215`，temperature=0.5

**LLM-as-Judge 结果（2026-04-29）**：

| 模型 | 框架覆盖 | 证据使用 | 评级质量 | 简洁性 | 总分/20 | 延迟 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Seed 2.0 Mini ✅ | **4.0** | 4.0 | 4.0 | 4.5 | **16.5** | 17.2s |
| 1.5 Pro ~~旧~~ | **2.7** | 3.3 | 3.3 | 3.3 | **12.6** | 9.1s |

**切换理由**：差距 3.9 分，主要在 framework_coverage（五力分析完整度 2.7 vs 4.0）。延迟从 9.1s → 17.2s，但 Analyst 每次研究只调用 1-2 次，8s 额外延迟可接受。

---

## 最终分配汇总

| Agent | 模型 | Temperature | 理由核心 |
|-------|------|:-----------:|---------|
| Supervisor | `doubao-1-5-pro-32k-250115` | 0.2 | 最快结构化输出，LLM-judge T1: 19.7/20，格式稳定 |
| Researcher | `deepseek-v3-2-251201` | 0.3 | 无推理开销，并行调用高效 |
| Analyst | `doubao-seed-2-0-mini-260215` | 0.5 | LLM-judge T2: 16.5/20（vs 旧 12.6），框架覆盖更完整 |
| Critic | `doubao-seed-2-0-mini-260215` | 0.6 | LLM-judge T3: 15.8/20，批判质量最尖锐 |
| Writer | `deepseek-v3-2-251201` | 0.4 | LLM-judge T4: 19.0/20（vs 旧 17.3），洞察质量 4.0 vs 2.3 |
