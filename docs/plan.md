# Hermes 任务花费预测 Agent：方案与分工

> 比赛：Hermes Agent Plugin Hackathon（Nous Research）
> 截止：**2026-11-11 5:00 PM EST（洛杉矶时间下午 2:00）**
> 仓库：`mr-jude-justin/hermes-plugin`（插件代码放在 `plugin/` 下）
> 团队：Justin `@mr-jude-justin`（lead，负责 Harness） · Eason `@easonluo727`（负责预测算法）

---

## 1. 一句话定位

**任务开始之前，先告诉你这次大概要花多少钱；跑的过程中一旦预计会超预算，就先停下来问你。**

要解决的痛点：用户因为不知道要花多少钱，就不敢让 Hermes 去做长线编程、生成图片这类大任务，用了贵的模型（比如 Opus）也不敢放手让它跑。

**只服务按 API 计费的用户（11/6 定）。** 用 Claude Pro/Max 等订阅、通过 OAuth 登录 Hermes 的用户不在支持范围内，原因有两个：
- Anthropic 的条款规定，订阅的 OAuth 登录只能用在 Claude Code 和 claude.ai 上，第三方工具要用 API key。
- 订阅只显示额度百分比，没有按次计算的美元花费，我们没有东西可以预测和对账。

这一点要写进插件 README 的风险说明和产品介绍里。

核心思路：**AI + 算法 → 做成一个运行很快的小 agent。** 光靠规则不够准，光靠大模型又慢又贵，也不准（见第 2 节的研究结论）。所以要用快速规则、用户自己的历史数据和一个便宜模型组成的 agent，三者配合来预测。

---

## 2. 现状调研：有没有人做过？

### 2.1 Hermes 自带的

| 功能 | 有没有 |
|---|---|
| `/usage`、`/insights`、Dashboard Analytics：事后查看 token 用量和估算费用（按模型分） | 有 |
| `/context`：开始前估算 token 数 | 有，但**不换算成美元** |
| 按美元设上限、超了就停 | **没有**（只能限制轮数，`goals.max_turns` 默认 20） |
| MCP 工具单次价格和每个对话的花费上限（`tool_costs` / `spend_limit`） | 有，但**只针对 MCP 工具，用积分计价，不包括模型本身的 token 费用** |
| **整个任务的事前估价** | **没有** |

### 2.2 Hermes 生态里已有的费用插件和工具

| 名称 | 在官方插件目录里吗 | 实时记账 | 超预算拦截 | **事前估价** |
|---|---|---|---|---|
| **hermes-telemetry**（nujovich）⚠️ 最直接的竞品 | **在** | 有（SQLite，本地价格表） | 有：默认每天 $5、每月 $100，用 `pre_tool_call` 拦截，80% 时提醒 | **没有** |
| agentguard-spend-hermes | 未说明 | 有 | 有（靠替换 SDK 的方法实现，**不符合目录"禁止覆盖核心代码"的规则**） | 没有 |
| hermes-stack / agent-cost-guard（Mukunda） | 不在，是外部 Python 库 | 有 | 有（调用之后才检查） | 没有 |
| SatGate | 不在，是一个网关 | 有 | 有（按每个工具的已知价格，在网关处拦截） | 只按单个工具的价格检查，**没有整个任务的预测** |

**结论：** "记账 + 超预算拦截"已经有人做了，而且 hermes-telemetry 就在官方目录里。**我们不能把这当卖点**，最多当作底层功能，或者直接和它配合使用。
**在 Hermes 上，"整个任务的事前估价"目前没人做。** 这是我们的差异点。

### 2.3 Hermes 之外的事前估价

| 名称 | 做什么 | 对我们的启发 |
|---|---|---|
| **tokencast** | "开始前估算 LLM agent 的花费，并自动校准"，以 Claude Code 插件或 MCP 的形式提供 | **思路最接近，但不是 Hermes 的**。它说明这个需求是真实的；我们要在 Hermes 上做，而且要做得更深（配合拦截和实时修正） |
| citadel-predict | 调用云端 API，给出 token 的期望值和上下限；只针对 claude-sonnet、20 类任务、80 次实测做过校准 | "给区间而不是给一个数"是对的 |
| **论文** *How Do AI Agents Spend Your Money?*（Stanford / Michigan / Google DeepMind 等，arXiv 2604.22750） | 研究编程 agent 的 token 消耗 | 见下 |

**论文里对算法最重要的几条结论（Eason 必读）：**
- **同一个任务跑多次，总 token 数最多能差 30 倍**，所以只给一个点估计必然不准，**一定要给区间**
- **成本主要来自输入 token，尤其是缓存读取**，不是输出。预测的重点是"会读多少上下文、跑多少轮"
- **直接让模型自己估，效果很差**：最好的结果，相关系数也只有 0.39，而且**系统性低估**。所以**不能只靠 LLM 自评**，必须用用户的历史数据来校准
- 准确率在中等花费时就到顶了，再多花钱也不会更准，所以"花费上限"本身就有意义

---

## 3. 系统架构

```
用户消息
   │
   ▼
[Harness] pre_llm_call ──► 收集特征 ──► [算法] 预测 ──► 返回预估区间和档次
   │                                              │
   │   低于门槛：直接放行（用户无感）               │
   │   高于门槛：block，并给出报价，等用户确认 ◄────┘
   ▼
主模型开始干活
   │
   ├─ 每次模型调用后 post_api_request ──► [Harness] 记账
   │                                       └─► [算法] 根据目前的消耗速度，重新预测最终总价
   │
   ├─ 每次工具调用前 pre_tool_call ──► 预计超过上限？是 → block，问用户继续还是停止
   │
   ▼
任务结束 on_session_end / post_llm_call ──► [算法] 比较预估和实际，校准，存入历史
```

两种运行模式（用户自己选）：
- **安全模式（默认）**：先估价再开工，多等约 1–2 秒，不会多花钱
- **快速模式**：`pre_llm_call` 在后台线程里估价、自己立刻放行；第一次 `pre_tool_call` 时检查估价结果。没有延迟，代价最多是第一步的钱

---

## 4. 分工

### 4.1 Eason：预测算法（核心）

负责把"特征"变成"预估区间"，并且越用越准。

1. **任务分档**：S / M / L / XL 四档，每档对应典型任务和花费区间（和 Justin 一起定标准）
2. **三层预测**
   - L1 **规则**（< 1 ms）：关键词（中英文）、消息长度、有没有代码块、路径、URL、附件、当前模型、会话里最近调用过哪些工具
   - L2 **历史比对**（约 10–50 ms）：在用户过去的任务里找最相似的几个（先用 TF-IDF 加余弦相似度，**不要引入 torch**），用它们的实际花费算区间
   - L3 **便宜模型**（约 1–2 s）：只处理前两层拿不准的情况，用 `ctx.llm` 指定便宜模型，让它判断档次和步数。**它的结果只当作一个参考信号，不直接当答案**（论文说明模型自评会低估）
3. **区间输出**：P50 和 P90，加上置信度
4. **跑到一半重新预测**：根据已经花的钱和每步的消耗速度，推算最终总价
5. **校准**：任务结束后比较预估和实际，修正各档次的系数
6. **冷启动**：没有历史数据时，用一份"常见任务的默认花费"作为初始值
7. **评估**：自己准备 20–30 个测试任务，记录预估和实际，算误差，demo 和 README 里要用

### 4.2 Justin：Harness（整个 agent 系统）

负责把算法需要的信息喂进去，把算法的结果落实成产品行为。

1. **Hook 接入**：`pre_llm_call`、`post_api_request`、`pre_tool_call`、`post_llm_call`、`on_session_end`、`subagent_start/stop`
2. **特征收集**：按第 5 节的接口，把消息、会话状态、模型、工具情况整理成特征，交给算法
3. **记账**：每次调用的 token 数（`usage`）乘以价格表，换算成美元
4. **价格表**：维护常用模型的输入、输出、缓存读取价格，以及图片生成等按次收费的价格（**不能引用 Hermes 内部的价格表**，审核时会被警告）
5. **预算策略**：单次上限、每日上限、触发报价的门槛、80% 提醒、100% 拦截。用 `config_schema` 声明，用户可以在 Desktop 设置页里改
6. **交互文案**：报价消息、拦截消息怎么写；用户怎么回复"继续""改上限""取消"
7. **估价 agent 能用的工具**：比如快速统计仓库有多少文件、多少行（帮助判断编程任务的规模），还有读取历史账本、读取当前会话状态
8. **存储**：用 `ctx.state` 或 `plugin_db()` 存账本和历史任务
9. **插件外壳**：`plugin.yaml`（声明 `llm.model_override` 权限、`provides_hooks`、依赖带上限版本）、README 风险说明、`hermes plugins validate` 通过

### 4.3 协作方式

- **Eason 需要什么信息，就去看板上 @ Justin。** 格式：
  `[Eason's Claude] @mr-jude-justin 需要新特征：<名字>，用途：<说明>，在哪个 hook 能拿到：<猜测>`
- Justin 加好后回复 @easonluo727，并更新第 5 节的接口
- **接口只增不改**，改字段要先在看板上说
- 文件归属：
  - Justin：`plugin/__init__.py`、`plugin/harness/`、`plugin/pricing.py`、`plugin/plugin.yaml`
  - Eason：`plugin/forecast/`（算法）、`plugin/forecast/tests/`、`eval/`（评估数据和脚本）

---

## 5. 接口约定（草案，第一天定下来）

**Harness → 算法：`TaskFeatures`**
```json
{
  "message": "帮我把这个项目重构成插件架构",
  "message_len": 18,
  "has_code_block": false,
  "has_paths_or_urls": false,
  "attachments": 0,
  "model": "claude-opus-5-5",
  "price": {"input_per_mtok": null, "output_per_mtok": null, "cache_read_per_mtok": null},
  "session_turns_so_far": 6,
  "context_tokens_now": 42000,
  "recent_tools": ["read_file", "terminal", "write_file"],
  "enabled_toolsets": ["terminal", "file", "web", "image_gen"],
  "cwd_is_git_repo": true,
  "repo_size": {"files": 312, "lines": 48000},
  "mode": "safe"
}
```

**算法 → Harness：`Estimate`**
```json
{
  "tier": "L",
  "cost_usd_p50": 2.2,
  "cost_usd_p90": 6.0,
  "est_steps": 25,
  "confidence": 0.6,
  "basis": "8 similar past tasks + rules",
  "layer_used": "L2",
  "estimate_cost_usd": 0.001
}
```

**跑到一半：`Reforecast`**（输入：已花费、已执行的步数、最近每步的花费；输出：预计最终总价的 P50 和 P90）

### 5.1 定稿（11/6 看板上 Justin 和 Eason 达成一致）

上面的 JSON 是草案，以本节为准。

- **算法预测 token 数，不直接预测美元。** `Estimate` 增加：
  - `tokens_p50` 和 `tokens_p90`，各自拆成 `{input, cache_read, output}`
  - `per_call_units`：按次收费的工作，例如 `{"image_gen": 3}`
  - `cost_usd_p50` 和 `cost_usd_p90` 保留，由 harness 用 `pricing.py` 换算后填入
- **Python 入口**（`plugin/forecast/`）：
  - `estimate(features: TaskFeatures) -> Estimate`
  - `reforecast(features, original: Estimate, spent: Usage, steps_done: int, recent_steps: list[Usage]) -> Estimate`
  - `record_outcome(features, estimate, actual: Usage, steps: int) -> None`
- **共享类型** `TaskFeatures`、`Estimate`、`Usage`、`PastTask` 统一放在 `plugin/contracts.py`，由 Justin 负责，接口只增不改。
- **存储由 harness 负责**：
  - `history.load() -> list[PastTask]`；`PastTask` 包含 `features`、`actual: Usage`、`steps`、`model`，以及当初的 `estimate`
  - 校准系数通过 `forecast_state.get(key)` / `forecast_state.set(key, value)` 读写
- **L3 便宜模型的花费也要记账**，金额取 `estimate_cost_usd`。

---

## 6. 今晚就要验证的事（不通过就要换方向）

- [ ] **最小测试插件**：在 `pre_llm_call` 里返回 `block`，回一句"预计 $X，继续吗？"，确认这样确实能**在主模型开始前**拦下来
- [ ] `pre_llm_call` 有没有超时限制？超时是放行还是拦下？
- [ ] `post_api_request` 里的 `usage` 字段长什么样？有没有缓存读取的 token 数？
- [ ] `ctx.llm.complete(model=...)` 能不能指定便宜模型？结果里的 `cost_usd` 准不准？
- [ ] 能不能在 hook 里开后台线程（快速模式需要）？
- [ ] **装上 hermes-telemetry 试用一下**：看清楚它做了什么，我们的插件要和它**配合**，而不是重复它
- [ ] UCLA 网络环境下，常用模型的 API 能不能正常调用（demo 需要真实花费）

---

## 7. 时间线

| 日期 | Justin（Harness） | Eason（算法） |
|---|---|---|
| **11/5（今晚）** | 跑通最小测试插件，完成第 6 节的验证 | 读论文，定分档标准初稿 |
| **11/6** | 定接口、hook 骨架、价格表、记账 | L1 规则 + 冷启动默认值 |
| **11/7** | 预算策略、报价和拦截交互 | L2 历史比对 + 区间计算 |
| **11/8** | 估价 agent 的工具（仓库规模等），把两边接起来 | L3 便宜模型 + 跑到一半重新预测 |
| **11/9** | 联调、`hermes plugins validate`、README、风险说明 | 校准 + 用 20–30 个任务做评估 |
| **11/10** | 录 90 秒 demo，仓库改成公开 | 把评估结果整理进 README |
| **11/11 上午** | pin commit → 给官方仓库提 PR → 在 X 上发视频 → 填 Typeform | 检查 |

---

## 8. Demo 脚本（90 秒，草案）

1. （0–15 秒）痛点："我接了 Opus，但不敢让它做大任务，因为不知道要花多少钱。"
2. （15–40 秒）发一个长线编程任务，**主模型开始前**弹出："L 级任务，预计 $2–6（参考你过去 8 次类似任务），超过你的单次上限 $3，继续吗？"
3. （40–65 秒）同意后开跑，实时计费；中途预测会超上限，在下一次工具调用前停下来问你
4. （65–80 秒）发一句闲聊，几乎没有延迟，也不打扰
5. （80–90 秒）收尾："hermes-telemetry 告诉你花了多少；我们告诉你**将要**花多少。"

---

## 9. 还没解决的问题

- ~~用订阅的用户（比如 Codex 订阅）在乎的是额度，不是美元，要不要兼顾？~~ 已定（11/6）：不兼顾，只服务 API 用户，原因见第 1 节。
- 估价不准的时候，怎么让用户还愿意信它？比如展示"上次预估 $2，实际 $2.4"
- 子 agent 和委派出去的任务，费用要怎么算进来？
- 和 hermes-telemetry 是直接读它的数据，还是各自记账？

---

## Sources

- [Hermes 插件开发文档](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins)
- [Hermes 插件目录提交规则](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins/catalog-submission)
- [Hermes slash commands](https://hermes-agent.nousresearch.com/docs/reference/slash-commands)
- [Hermes CLI commands](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)
- [Hermes web dashboard](https://hermes-agent.nousresearch.com/docs/user-guide/features/web-dashboard)
- [hermes-telemetry（官方目录）](https://hermes-agent.nousresearch.com/docs/plugins/hermes-telemetry)
- [hermes-telemetry 介绍文章](https://dev.to/nujovich/a-plugin-for-observability-budget-guardrails-built-with-hermes-agent-4b63)
- [agentguard-spend-hermes](https://pypi.org/project/agentguard-spend-hermes)
- [hermes-stack](https://dev.to/mukundakatta/hermes-stack-four-governance-layers-for-hermes-agent-in-200-lines-3n22)
- [agent-cost-guard](https://dev.to/mukundakatta/my-hermes-agent-spent-3-before-i-noticed-now-it-cant-37dk)
- [SatGate for Hermes Agent](https://satgate.io/satgate-for-hermes-agent)
- [tokencast](https://ai.g2.com/marketplace/tools/tokencast)
- [citadel-predict](https://pypi.org/project/citadel-predict/)
- [How Do AI Agents Spend Your Money?（arXiv 2604.22750）](https://arxiv.org/abs/2604.22750)
