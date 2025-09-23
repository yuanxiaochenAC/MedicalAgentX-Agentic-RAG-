# 🏥 MedicalAgentX：医疗智能 Agentic RAG 系统

## 📖 项目简介

MedicalAgentX是一个基于多Agent协作的智能医疗分析系统，结合了向量数据库检索（FAISS）、大语言模型推理（LLM）和结构化医学工具（ToolUniverse），为医疗专业人员提供AI辅助的病例分析和诊断支持。

### 🎯 核心特性

- **🔍 语义检索**：使用FAISS向量数据库快速检索相似医学病例
- **🤖 多Agent协作**：基于EvoAgentX的智能Agent编排和执行
- **🧠 医学推理**：结合大语言模型进行专业的病因分析
- **🔧 工具增强**：集成ToolUniverse医学工具获取结构化知识
- **📋 综合报告**：生成结构化的医学分析报告

### ⚠️ 重要声明

**本系统仅供医疗专业人员参考使用，不能替代正式的医疗诊断。最终诊断必须由具有资质的医师结合完整的临床信息做出。**

## 🏗️ 系统架构

```
MedicalAgentX/
├── app/                     # 主程序模块
│   ├── config.py           # 系统配置
│   ├── ingestion.py        # PDF文档导入与索引
│   └── evo_agent_runner.py # Agent工作流执行器
├── agents/                  # Agent配置文件
│   ├── retriever_agent.yaml      # 检索Agent
│   ├── reasoning_agent.yaml      # 推理Agent  
│   ├── tool_universe_agent.yaml  # 工具Agent
│   └── integration_agent.yaml    # 综合Agent
├── workflows/               # 工作流定义
│   └── medical_case_analysis.yaml
├── data/                    # 数据存储
│   └── medical_cases/       # PDF病例文档
└── outputs/                 # 输出结果
    ├── logs/               # 系统日志
    └── results/            # 分析结果
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目（假设已有EvoAgentX和ToolUniverse）
cd MedicalAgentX

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

确保在项目根目录有以下文件：
- `openai_api_key.txt`：包含OpenAI API密钥
- 其他所需的API配置文件

### 3. 导入医学文档

```bash
# 将PDF医学案例文档放入 data/medical_cases/ 目录
# 运行文档导入和索引创建
cd app
python ingestion.py
```

### 4. 运行医学分析

```python
from app.evo_agent_runner import MedicalAgentExecutor

# 创建执行器实例
executor = MedicalAgentExecutor()

# 分析病例
symptom_text = "患者男性，58岁，体重下降，黄疸，上腹胀痛，ALT升高"
results = executor.run_medical_analysis_workflow(symptom_text)

# 查看分析报告
print(results["final_report"])
```

## 📊 工作流程

### 1. 检索阶段 (RetrieverAgent)
- 使用FAISS向量数据库检索语义相似的医学病例
- 返回最相关的案例文档片段

### 2. 推理阶段 (ReasoningAgent)  
- 基于检索的病例进行医学推理
- 分析症状、可能病因、鉴别诊断
- 提供检查建议和风险评估

### 3. 工具查询阶段 (ToolUniverseAgent)
- 调用医学知识工具获取结构化信息
- 查询药物信息、诊断标准、治疗指南

### 4. 综合报告阶段 (IntegrationAgent)
- 整合所有分析结果
- 生成完整的医学分析报告
- 包含诊断建议、治疗方案、随访计划

## 🔧 配置说明

### FAISS配置
```python
FAISS_CONFIG = {
    "corpus_id": "medical_case_reports",
    "embedding_model": "text-embedding-ada-002",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5
}
```

### LLM配置
```python
LLM_CONFIG = {
    "model": "gpt-4",
    "temperature": 0.3,
    "max_tokens": 2000
}
```

## 📝 使用示例

### 基础使用
```python
# 导入必要模块
from app.evo_agent_runner import MedicalAgentExecutor

# 创建执行器
executor = MedicalAgentExecutor()

# 分析症状
symptoms = "患者反复头痛伴恶心呕吐，视物模糊，血压升高"
result = executor.run_medical_analysis_workflow(symptoms)

# 获取结果
if result["status"] == "success":
    print("🩺 医疗分析报告:")
    print(result["final_report"])
else:
    print(f"❌ 分析失败: {result['error']}")
```

### 高级配置
```python
# 自定义分析深度
result = executor.run_medical_analysis_workflow(
    symptom_text=symptoms,
    analysis_depth="comprehensive",  # basic/standard/comprehensive
    patient_info={
        "age": 45,
        "gender": "female",
        "medical_history": ["高血压", "糖尿病"]
    }
)
```

## 🔍 支持的医学专科

- 内科 (Internal Medicine)
- 外科 (Surgery)  
- 神经科 (Neurology)
- 心血管科 (Cardiology)
- 消化科 (Gastroenterology)
- 呼吸科 (Pulmonology)
- 内分泌科 (Endocrinology)
- 血液科 (Hematology)
- 肿瘤科 (Oncology)
- 风湿免疫科 (Rheumatology)

## 📈 性能优化

### 向量数据库优化
- 使用适当的chunk_size和overlap
- 定期更新和维护FAISS索引
- 考虑使用GPU版本的FAISS提升检索速度

### LLM调用优化
- 设置合理的温度参数确保医学准确性
- 使用缓存机制减少重复调用
- 实现请求限流避免API超限

## 🛠️ 开发指南

### 添加新的Agent
1. 在`agents/`目录创建新的YAML配置文件
2. 定义Agent的输入输出参数
3. 在工作流中引用新Agent

### 扩展医学工具
1. 在ToolUniverse中添加新的医学工具
2. 更新`tool_universe_agent.yaml`配置
3. 修改工具调用逻辑

### 自定义工作流
1. 创建新的workflow YAML文件
2. 定义Agent执行顺序和依赖关系
3. 配置错误处理和质量控制

## 🧪 测试

```bash
# 运行单元测试
pytest tests/

# 运行集成测试
pytest tests/integration/

# 生成测试报告
pytest --cov=app tests/
```

## 📊 监控和日志

### 日志配置
- 系统日志存储在`outputs/logs/`
- 包含详细的Agent执行信息
- 支持不同级别的日志输出

### 性能监控
- 跟踪工作流执行时间
- 监控API调用频率和成功率
- 记录检索质量指标

## 🔒 安全和隐私

### 数据保护
- 患者信息在处理过程中匿名化
- 日志文件不包含敏感医疗信息
- API密钥安全存储

### 访问控制
- 实现基于角色的访问控制
- 审计所有医疗分析操作
- 确保符合医疗数据保护法规

## 🤝 贡献指南

1. Fork项目仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🆘 支持和帮助

### 常见问题
- **Q: 如何添加新的医学文档？**
  A: 将PDF文件放入`data/medical_cases/`目录，然后运行`python app/ingestion.py`重新构建索引。

- **Q: 分析结果不准确怎么办？**  
  A: 检查输入症状描述是否详细，确保有足够的相似病例，可以调整检索参数或添加更多训练数据。

- **Q: 如何自定义医学专科？**
  A: 修改`config.py`中的`MEDICAL_CONFIG`，添加新的专科和关键词。

### 技术支持
- 提交Issue到GitHub仓库
- 联系开发团队：medical-ai-team@example.com
- 查看文档：[项目Wiki](wiki-link)

---

**⚠️ 医疗免责声明**

本系统提供的分析结果仅供医疗专业人员参考，不能替代正式的医疗诊断。任何医疗决策都应该基于完整的临床评估，并遵循相关的医疗指南和标准。如有紧急医疗情况，请立即寻求专业医疗帮助。

---

*MedicalAgentX - 让AI为医疗专业人员提供智能决策支持* 🏥✨