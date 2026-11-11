"""Editorial pass applied to the raw sessions before they are shown on the demo page.

The demo is for people learning what DecisionForge can do, so the playback keeps the task
itself and removes repository housekeeping (issue/PR bookkeeping, directory tours,
storage cleanup, session summaries). User messages are rewritten as plain requirements.

Keys are UTC timestamp prefixes of the original events (see the export). ``None``
means "drop this event". Facts and numbers in rewritten agent messages come from
the original messages at the same timestamp.
"""

# Explicit migration map for the original session commands. Ordered longest first.
# This table intentionally retains historical spellings so regeneration is reproducible.
COMMAND_REPLACEMENTS = [
    ("python run.py train --validate-only", "uv run dforge train validate"),
    ("python run.py eval --score-only", "uv run dforge eval rescore"),
    *[(f"python run.py {name}", f"uv run dforge data {name}") for name in ("parse-pdf", "inspect", "build", "resume", "validate", "export")],
    ("python run.py eval-list", "uv run dforge eval list"),
    ("python run.py eval", "uv run dforge eval run"),
    ("python run.py train", "uv run dforge train run"),
    ("python run.py schemas", "uv run dforge schema export"),
    ("python run.py", "uv run dforge"),
    ("AUTOJEV_", "DFORGE_"),
    ("AutoJev", "DecisionForge"),
    ("src/autojev", "src/decisionforge"),
    ("from autojev", "from decisionforge"),
    ("import autojev", "import decisionforge"),
    ("autojev-data", "dforge-data"),
    ("autojev-train", "dforge-train"),
    ("autojev-eval", "dforge-eval"),
    ("autojev-analyze", "dforge-analyze"),
    ("Autojev-data", "DecisionForge"),
    ("Autojev", "DecisionForge"),
    ("AutoJEV-Data", "DecisionForge"),
    ("AutoJEV", "DecisionForge"),
    ("autojev", "dforge"),
    *[(f"uv run dforge {name}", f"uv run dforge data {name}") for name in ("parse-pdf", "inspect", "build", "resume", "validate", "export")],
]
REPLACE = [("PR #13", "参考实现"), *COMMAND_REPLACEMENTS]

TRIAGE = {
    "end": "2026-10-29T02:22:00",
    "sessions": {1: "会话 1 · 明确任务需求", 2: "会话 2 · 执行：数据、训练、评测与优化"},
    "drop_windows": [
        ("2026-10-28T05:24:00", "2026-10-28T05:27:00"),  # 精简需求文档的来回
        ("2026-10-28T06:53:40", "2026-10-28T06:54:03"),  # 询问工作目录
        ("2026-10-28T09:49:00", "2026-10-28T09:49:59"),  # 再次列目录
        ("2026-10-28T09:58:00", "2026-10-28T10:08:05"),  # 共享环境被手动清理的插曲
    ],
    "rewrite": {
        "2026-10-28T05:17:05": (
            "参照仓库里电网示例的形式，做一个医疗分诊场景：分诊决策 + 患者消息优先级。\n\n"
            "评测只用公开 benchmark：CARE-Bench、PMR-Synth 标注对、PMR-Reddit；训练数据由我们自己搜集资料、构造。"
            "底座用 Qwen3.5-9B，主要用 Unsloth LoRA，做出尽可能好的模型，并与 Jev、LLM2Jev、AnyJev、Chat 对比。"
            "先帮我把任务需求和评测口径定下来。"
        ),
        "2026-10-28T05:17:12": "我先梳理分诊与消息优先级两类任务的边界和评测口径，同时把仓库更新到最新，核对电网示例和此前的医疗实验，再整理成一份可执行的任务需求。",
        "2026-10-28T05:17:49": "此前做过一次 4B 模型实验，但数据构造和评测范围与这次不同。另一个关键点是：CARE-Bench 评的是逐轮信息披露后的分诊动作，PMR 评的是两条消息谁更紧急，需要分别定义指标。",
        "2026-10-28T05:20:29": (
            "已确认采用三套完整公开评测集作为主结果。需求聚焦分诊和消息优先级：训练数据全部自建，不沿用旧方案的 CARE/PMR 训练划分和 ESI 辅助任务；"
            "同时明确自训模型与 Jev、LLM2Jev、AnyJev、Chat 的比较条件。\n\n需求写好后先做一次独立审阅，重点检查数据泄漏和验收标准。"
        ),
        "2026-10-28T05:23:15": (
            "需求通过独立审阅，未发现阻塞问题。\n\n其中明确了两个容易造成虚高分的问题：Reddit 的 `chosen/rejected` 字段不能直接给模型；"
            "默认 generic 评分的 argmax 和分母规则，也需要调整后才能满足这次的评测口径。"
        ),
        "2026-10-28T05:23:52": (
            "任务需求已确定：\n\n"
            "- **训练**：独立搜集资料、构造数据，经 DecisionForge 流程训练 Qwen3.5-9B，主要采用 Unsloth LoRA。\n"
            "- **评测**：CARE Public Test 1 全部 135 个前缀、PMR-Synth 869 对、PMR-Reddit 1,502 对，仅用于评测。\n"
            "- **对照**：自训模型与 Jev、LLM2Jev、AnyJev L0、Chat。\n"
            "- **最优模型**：在约定预算内，用自建验证集选优，再冻结模型进行正式测试。\n\n"
            "需求包含防泄漏要求、评分规则和验收标准。本轮只完成需求定义，尚未启动生成或训练；执行时再确定算力、API 服务和预算。"
        ),
        "2026-10-28T05:27:43": "按 dforge skill 开始执行：先核对任务需求、已有产物和当前作业状态，再推进分诊模型训练，并检查 Jev 调用是否需要走代理。",
        "2026-10-28T06:49:43": "评测的转换和评分代码，请封装成任务目录下的 benchmark 扩展。现在是这样放的吗？",
        "2026-10-28T07:04:29": None,
        "2026-10-28T10:27:02": "模型、15 项原始结果和复现资料都已核对。接下来释放本任务申请的 H200 作业；源码、benchmark 扩展、模型和报告都会保留。",
        "2026-10-28T10:39:18": (
            "训练及全部 **15 项评测已完成**，完整报告在任务目录的 `evaluation/comparison-001/RESULTS.md`；"
            "全仓测试 **945 通过、48 跳过、3 项既有失败**。\n\n"
            "<details>\n<summary>执行中的取舍（6 项）</summary>\n\n"
            "1. 评测代码按要求封装为任务目录内的 benchmark 扩展，各评测环境都需安装。\n"
            "2. 如实报告既有测试失败，不宣称全仓全绿。\n"
            "3. 逐项核验实际模型和完整结果，代码审查不代替运行验收。\n"
            "4. 不宣称临床有效性、语义独立或预训练独立。\n"
            "5. 保留提前启动 LLM2Jev 的顺序偏差，不宣称严格盲测。\n"
            "6. AnyJev 用新运行修复兼容性，旧失败和额外计算开销都保留在记录里。\n\n"
            "</details>"
        ),
        "2026-10-28T10:50:36": "Reddit 上的效果还差一点，还有哪些可以优化的地方？",
        "2026-10-28T10:59:05": "可以，按这个方向继续优化。",
        "2026-10-29T02:17:29": "这轮优化没有带来提升，最终结果采用 v1 版本。",
    },
    "questions": [
        ("新 issue 的主评测", "这次的主评测"),
        ("沿用 PR #13 里的小题集", "沿用参考实现里的小题集"),
    ],
    "edits": {
        "2026-10-28T14:17:37": [("任务环境未引用旧 issue20 路径。", "")],
    },
    "answers": {
        "rjob pod": "用集群 GPU 任务（rjob pod）",
        "。env 里面又 boyue 的api，我们用 deepseek-flash": "用 .env 里已配置的接口，模型选 deepseek-flash",
        "用的是 openrouter 的，我们有 openrouter 的 api": "Jev 走 OpenRouter，用现有的 OpenRouter API",
    },
    "subagents": {
        "review_triage_issue": ("需求审阅", "需求通过独立审阅：数据泄漏约束与验收标准完整，未发现阻塞问题。"),
        "design_review": ("执行方案审阅", "执行方案通过独立审阅：数据分组留出、独立验证选模、校准与全量评测的顺序没有影响实施的缺口。"),
        "implementation_review": (
            "实现审阅",
            "只读审阅完整分支改动与两个编排脚本，49 项 CPU 测试通过，未调用付费接口或 GPU。\n\n"
            "- 优点：benchmark 转换不泄漏标签，换位时病史、消息和金标同步；指标保留原生 Choice 与完整分母；数据组装保守隔离来源与患者重叠。\n"
            "- 发现两处完整性问题：中断窗口可能让教师接口重复提交已有响应；对比门槛只核对模型路径、没有核对文件哈希。两处均由主 agent 补回归测试后修复。",
        ),
        "optimization_review": (
            "优化方案审阅",
            "发现 1 个 P2 问题：冻结记录保护触发补数时，控制器会重新请求全部数据角色；训练集达到轮次上限后，可能挡住验证/校准集的补数。"
            "该路径会安全失败、不会混入污染数据。主 agent 随后修复并补了回归测试。",
        ),
    },
}

MATERIALS = {
    "end": "2026-11-02T11:30:00",
    "sessions": {1: "会话 1 · 明确任务需求", 2: "会话 2 · 执行：评测集、数据、训练与评测"},
    "drop_windows": [
        ("2026-10-30T00:00:00", "2026-11-02T07:37:59"),  # 找本地文件、整理示例数据、清理存储
    ],
    "rewrite": {
        "2026-10-29T08:18:14": (
            "我们有一批材料性能预测的数据和参考答案（eval_760.jsonl），把它作为评测集，需要先构造选项和 context。\n\n"
            "训练数据请用 DecisionForge 的功能自行搜寻、构造选项。底座 Qwen3.5-9B、主要用 Unsloth LoRA，与 Jev、LLM2Jev、AnyJev、Chat 对比，"
            "其余设置和分诊任务保持一致。先帮我把任务需求整理清楚。"
        ),
        "2026-10-29T08:26:05": None,
        "2026-10-29T08:39:28": None,
        "2026-10-29T08:42:24": None,
        "2026-10-29T08:42:59": "可以，就按这份需求执行。",
        "2026-10-29T08:43:32": None,
        "2026-10-29T08:46:02": (
            "/dforge 按这份需求完成材料性能预测任务，训练出比 Jev 更强的模型。\n\n"
            "GPU 通过集群 rjob 申请；Jev 走 OpenRouter，需要挂代理；尽量直接用仓库自带的命令；API key 等配置在 .env 中。"
        ),
        "2026-10-29T08:46:05": "我先读取任务需求和 DecisionForge skills 的使用说明。",
        "2026-10-29T09:16:03": "需求和关键决定都已确认，开始建立任务工作目录。",
        "2026-10-29T10:33:30": None,
        "2026-10-29T10:36:44": "GPU 一就绪就开始训练。",
        "2026-10-29T11:14:23": "为什么单卡也排不上？",
        "2026-11-02T07:38:06": "再找 1–2 个公开的材料 benchmark，补充评测一下。",
        "2026-11-02T08:29:26": None,
        "2026-11-02T08:29:40": "公开 benchmark 上也请把 Chat、LLM2Jev、AnyJev 和 Jev 都测一下；Jev 需要走代理。",
        "2026-11-02T08:40:59": "Jev 的请求要显式指定境外代理节点，集群默认代理出不去。",
        "2026-11-02T09:17:45": "等 GPU 任务跑完，把五方法结果表补全。",
    },
    "edits": {
        "2026-10-29T08:42:43": [
            ("已精简，保留 issue #22 的结构和篇幅。全文如下（草稿在 `$SHARED/scratch/issue-materials-property-draft.md`）：", "我把需求整理成下面这份文档："),
            ("参考背景：issue #22、PR #24。\n", ""),
            ("；最终按 issue #22 的格式追加结果与方法/资源消耗链接", ""),
            ("这样长度和 issue #22 接近了。确认的话我就创建到 `Oscar-dzy/Autojev-data`；", "确认后就按这份需求开始执行；"),
        ],
        "2026-11-02T09:06:05": [
            ("issue #25 的 760 题领先", "760 题上的领先"),
            ("跑完我会补齐五方法表并更新 PR。", "跑完我会补齐五方法表。"),
        ],
        "2026-11-02T09:20:30": [("README/PR", "README")],
        "2026-11-02T10:24:11": [("更新五方法表和 PR", "更新五方法表")],
    },
    "questions": [
        ("确认上述选择后，issue 是先出草稿给你看，还是直接创建到 GitHub？", "确认上述选择后，需求是先出草稿给你看，还是直接定稿？"),
        ("issue #25 要求", "需求要求"),
        ("issue 只规定", "需求只规定"),
        ("发布方式", "需求定稿"),
        ("直接创建 GitHub issue", "直接定稿"),
        ("先把 issue 正文写入本地文件并在对话里给你看，确认内容和选项协议后再创建到 GitHub。", "先把需求写成草稿给你看，确认内容和选项协议后再定稿。"),
        ("按你的描述和我的默认协议直接创建到 Oscar-dzy/Autojev-data，并把链接给你。", "按你的描述和我的默认协议直接定稿。"),
        ("（沿用 #22）", "（与分诊任务相同）"),
        ("沿用 #22 的", "沿用分诊任务的"),
    ],
    "answers": {},
    "subagents": {},
}

CURATION = {"triage": TRIAGE, "materials": MATERIALS}
