const skills = [
  { name: 'dforge', title: '从目标开始', detail: '澄清需求、判断适用性、确认路线，也可以接续已有任务。', example: '/dforge 我想做消息优先级判断，目前只有场景、没有数据。请先展示需求澄清方案和待回答的问题，等待我接受、修改或暂停。任务目录：outputs/tasks/message-priority/。' },
  { name: 'dforge-data', title: '研究与准备数据', detail: '找来源、整理材料、生成 JEV 数据，并检查质量与数据隔离。', example: '/dforge-data 帮我寻找适合消息优先级判断的数据集和原始材料。先展示资料搜寻方案，说明来源范围、筛选标准和停止条件，等待确认后执行。' },
  { name: 'dforge-train', title: '训练与恢复', detail: '确认模型、数据、方法与资源，再启动、跟进或恢复训练。', example: '/dforge-train 我想训练决策模型。请先核查我提供的数据、模型与资源，展示本阶段方案并等待确认；缺少信息先问，不启动训练。' },
  { name: 'dforge-eval', title: '评测与比较', detail: '评测已有模型、比较结果，也可以复用已有预测重评分。', example: '/dforge-eval 我只想比较已有模型。请先核查模型、评测集、指标和资源，展示评测方案并等待确认；缺少信息先问，不启动评测。' },
  { name: 'dforge-analyze', title: '分析与建议', detail: '解读数据、训练和评测证据，交付报告与下一步建议。', example: '/dforge-analyze 我想分析已有评测结果。请先确认报告路径、分析维度和交付物，展示分析方案并等待确认，不重新推理或训练。' },
];
const workflow = [
  ['澄清需求', '目标、选项与评价标准'],
  ['确认路线', '数据条件、资源与终点'],
  ['准备数据', '来源、划分与质量检查'],
  ['选择是否训练', '训练模型 / 使用已有模型'],
  ['评测与对比', '独立评测集与统一指标'],
  ['分析报告', '证据、边界与下一步建议'],
];
const resumeExample = '/dforge 继续 outputs/tasks/my-task/。先读取 README.md 和 task.json，核查实际 manifest、进程/作业及产物；展示恢复后的配置和待决定项，沿用仍有效的授权，不重复提交已有作业。';
const inspectExample = '/dforge 只做接入预检：输出实际 pwd 和 python 的 sys.executable，读取 README.md，再执行 OPENAI_MODEL=offline-inspect uv run dforge data inspect --config configs/default.yaml。这些只读预检已授权，不执行生成、训练或评测业务。';

const en = {
  skills: [
    { name: 'dforge', title: 'Start from a goal', detail: 'Clarify requirements, assess suitability, confirm a route or resume a task.', example: '/dforge I want to prioritize messages and only have a scenario. Show the requirements plan and open questions first. Task directory: outputs/tasks/message-priority/.' },
    { name: 'dforge-data', title: 'Research and prepare data', detail: 'Find sources, prepare materials, generate JEV records and check quality and data isolation.', example: '/dforge-data Find datasets and primary sources for message prioritization. Show source scope, selection criteria and stopping conditions before execution.' },
    { name: 'dforge-train', title: 'Train and resume', detail: 'Confirm model, data, method and resources before starting or resuming training.', example: '/dforge-train Check my data, model and resources. Show a stage plan and wait for confirmation before training.' },
    { name: 'dforge-eval', title: 'Evaluate and compare', detail: 'Evaluate existing models, compare results or rescore saved predictions.', example: '/dforge-eval Compare my existing models. Check the evaluation set, metrics and resources; show the plan before execution.' },
    { name: 'dforge-analyze', title: 'Analyze and recommend', detail: 'Interpret data, training and evaluation evidence; deliver a report and next steps.', example: '/dforge-analyze Analyze my evaluation results. Confirm report paths, dimensions and outputs. Reuse saved results without rerunning inference or training.' },
  ],
  workflow: [['Clarify requirements', 'Goals, options and criteria'], ['Confirm route', 'Data, resources and endpoint'], ['Prepare data', 'Sources, splits and quality'], ['Choose whether to train', 'Train or use an existing model'], ['Evaluate and compare', 'Independent sets and shared metrics'], ['Analyze results', 'Evidence, limits and next steps']],
  resumeExample: '/dforge Continue outputs/tasks/my-task/. Read README.md and task.json, check manifests, processes/jobs and outputs. Show recovered configuration and pending decisions; retain valid authorization and avoid duplicate jobs.',
  inspectExample: '/dforge Read-only preflight: show pwd and Python sys.executable, read README.md, then run OPENAI_MODEL=offline-inspect uv run dforge data inspect --config configs/default.yaml. These checks are authorized; do not generate data, train or evaluate.',
};
export const content = { zh: { skills, workflow, resumeExample, inspectExample }, en };
