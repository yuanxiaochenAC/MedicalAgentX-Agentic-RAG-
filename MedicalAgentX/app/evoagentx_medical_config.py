"""
真正的EvoAgentX医疗智能系统配置
True EvoAgentX Medical Intelligence System Configuration
"""

import os
from pathlib import Path
from typing import Dict, Any, List

from evoagentx.models import OpenAILLMConfig
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, IndexConfig, EmbeddingConfig, RetrievalConfig
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig

# 项目配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "medical_cases"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 确保目录存在
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def load_api_key() -> str:
    """加载OpenAI API密钥"""
    api_key_file = PROJECT_ROOT.parent / "openai_api_key.txt"
    try:
        with open(api_key_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key文件未找到: {api_key_file}")

# OpenAI LLM配置
OPENAI_API_KEY = load_api_key()
LLM_CONFIG = OpenAILLMConfig(
    model="gpt-4",
    openai_key=OPENAI_API_KEY,
    stream=True,
    output_response=True,
    temperature=0.3,
    max_tokens=2000
)

# EvoAgentX存储配置
STORAGE_CONFIG = StoreConfig(
    dbConfig=DBConfig(
        db_name="sqlite",
        path=str(CACHE_DIR / "medical_cases.sql")
    ),
    vectorConfig=VectorStoreConfig(
        vector_name="faiss",
        dimensions=1536,  # text-embedding-ada-002的维度
        index_type="flat_l2",
    ),
    graphConfig=None,
    path=str(CACHE_DIR / "indexing")
)

# EvoAgentX RAG配置
RAG_CONFIG = RAGConfig(
    reader=ReaderConfig(
        recursive=False, 
        exclude_hidden=True,
        num_files_limit=None, 
        custom_metadata_function=None,
        extern_file_extractor=None,
        errors="ignore", 
        encoding="utf-8"
    ),
    chunker=ChunkerConfig(
        strategy="simple",
        chunk_size=1000,
        chunk_overlap=200,
        max_chunks=None
    ),
    embedding=EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-ada-002",
        api_key=OPENAI_API_KEY,
    ),
    indexing=IndexConfig(
        name="medical_cases_index",
        index_type="flat_l2",
        metadata_to_index=[],
        metadata_config={}
    ),
    retrieval=RetrievalConfig(
        retrivel_type="vector",
        postprocessor_type="simple", 
        top_k=5,
        similarity_cutoff=0.1,  # 降低相似度阈值，从默认的很高值降到0.1
        keyword_filters=None,
        metadata_filters=None
    )
)

# 医疗工作流任务定义
MEDICAL_WORKFLOW_TASKS = [
    {
        "name": "MedicalRetriever",
        "description": "从医学文档库中检索相似病例",
        "inputs": [
            {
                "name": "symptom_text", 
                "type": "str", 
                "required": True, 
                "description": "患者症状描述"
            },

        ],
        "outputs": [
            {
                "name": "similar_cases", 
                "type": "str", 
                "required": True, 
                "description": "检索到的相似病例"
            },
            {
                "name": "retrieval_metadata", 
                "type": "str", 
                "required": True, 
                "description": "检索元数据信息"
            }
        ],
        "prompt": """作为医学文档检索专家，你需要分析患者症状并提供相关的医学案例检索。

患者症状描述：
{symptom_text}

请基于给定的症状描述，提供医学案例检索分析：
1. 症状关键词识别
2. 相关医学领域分析
3. 潜在诊断方向
4. 建议检索策略

以结构化格式输出检索分析。""",
        "parse_mode": "str"
    },
    {
        "name": "MedicalReasoner", 
        "description": "基于相似病例进行医学推理分析",
        "inputs": [
            {
                "name": "symptom_text", 
                "type": "str", 
                "required": True, 
                "description": "患者症状描述"
            },
            {
                "name": "similar_cases", 
                "type": "str", 
                "required": True, 
                "description": "检索到的相似病例"
            },
            {
                "name": "retrieval_metadata", 
                "type": "str", 
                "required": True, 
                "description": "检索元数据信息"
            }
        ],
        "outputs": [
            {
                "name": "medical_analysis", 
                "type": "str", 
                "required": True, 
                "description": "医学分析结果"
            },
            {
                "name": "primary_diagnoses", 
                "type": "str", 
                "required": True, 
                "description": "主要诊断方向"
            },
            {
                "name": "recommended_tests", 
                "type": "str", 
                "required": True, 
                "description": "建议检查项目"
            }
        ],
        "prompt": """你是一位资深的临床医生，请基于患者症状和相似病例进行专业的医学分析。

患者症状：
{symptom_text}

相似病例分析：
{similar_cases}

检索元数据：
{retrieval_metadata}

请提供详细的医学分析，包括：

## 症状分析
- 主要症状特征和临床意义
- 伴随症状的重要性
- 症状发展模式

## 可能病因
- 最可能的3-5个诊断方向
- 每个诊断的支持证据
- 病理生理机制分析

## 鉴别诊断
- 需要排除的疾病
- 鉴别要点和关键特征

## 检查建议
- 必要的实验室检查
- 影像学检查建议
- 特殊检查项目

## 风险评估
- 病情严重程度
- 可能的并发症
- 紧急程度评估

请提供循证医学的专业分析。""",
        "parse_mode": "str"
    },
    {
        "name": "MedicalToolsConsultant",
        "description": "调用医学工具获取补充信息",
        "inputs": [
            {
                "name": "medical_analysis", 
                "type": "str", 
                "required": True, 
                "description": "医学分析结果"
            },
            {
                "name": "primary_diagnoses", 
                "type": "str", 
                "required": True, 
                "description": "主要诊断方向"
            }
        ],
        "outputs": [
            {
                "name": "tool_consultation", 
                "type": "str", 
                "required": True, 
                "description": "工具咨询结果"
            },
            {
                "name": "additional_guidance", 
                "type": "str", 
                "required": True, 
                "description": "额外指导建议"
            }
        ],
        "prompt": """作为医学知识库专家，请基于分析结果提供补充的医学指导。

医学分析结果：
{medical_analysis}

主要诊断方向：
{primary_diagnoses}

请提供以下方面的专业建议：

## 药物指导
- 相关治疗药物选择
- 用药注意事项
- 可能的药物相互作用

## 诊断标准
- 相关疾病的诊断标准
- 临床指南参考
- 最新研究进展

## 治疗建议
- 标准治疗方案
- 个体化治疗考虑
- 预后评估

## 预防措施
- 疾病预防要点
- 生活方式建议
- 随访计划

请提供基于循证医学的专业建议。""",
        "parse_mode": "str"
    },
    {
        "name": "MedicalReportGenerator",
        "description": "生成综合医学分析报告",
        "inputs": [
            {
                "name": "symptom_text", 
                "type": "str", 
                "required": True, 
                "description": "患者症状描述"
            },
            {
                "name": "similar_cases", 
                "type": "str", 
                "required": True, 
                "description": "相似病例分析"
            },
            {
                "name": "medical_analysis", 
                "type": "str", 
                "required": True, 
                "description": "医学分析结果"
            },
            {
                "name": "primary_diagnoses", 
                "type": "str", 
                "required": True, 
                "description": "主要诊断方向"
            },
            {
                "name": "recommended_tests", 
                "type": "str", 
                "required": True, 
                "description": "建议检查项目"
            },
            {
                "name": "tool_consultation", 
                "type": "str", 
                "required": True, 
                "description": "工具咨询结果"
            },
            {
                "name": "additional_guidance", 
                "type": "str", 
                "required": True, 
                "description": "额外指导建议"
            }
        ],
        "outputs": [
            {
                "name": "comprehensive_report", 
                "type": "str", 
                "required": True, 
                "description": "完整的医学分析报告"
            },
            {
                "name": "executive_summary", 
                "type": "str", 
                "required": True, 
                "description": "执行摘要"
            }
        ],
        "prompt": """作为资深医学报告专家，请整合所有分析信息生成一份完整的医学分析报告。

## 输入信息

### 患者症状
{symptom_text}

### 相似病例分析
{similar_cases}

### 医学分析结果
{medical_analysis}

### 主要诊断方向
{primary_diagnoses}

### 建议检查项目
{recommended_tests}

### 工具咨询结果
{tool_consultation}

### 额外指导建议
{additional_guidance}

---

请生成一份结构完整的医学分析报告：

# 医学综合分析报告

## 执行摘要
- 病例概述
- 主要发现
- 关键建议

## 1. 病例基本信息
- 症状描述
- 临床表现特点

## 2. 相似病例参考
- 检索病例匹配情况
- 相似度分析
- 参考价值评估

## 3. 临床分析
### 3.1 症状分析
### 3.2 可能诊断
### 3.3 鉴别诊断

## 4. 检查建议
### 4.1 必需检查
### 4.2 选择性检查

## 5. 治疗指导
### 5.1 药物治疗
### 5.2 非药物治疗
### 5.3 生活指导

## 6. 风险管理
### 6.1 并发症预防
### 6.2 随访计划

## 7. 专业建议
### 7.1 临床决策支持
### 7.2 进一步咨询建议

## 免责声明
本报告基于AI辅助分析，仅供医疗专业人员参考。最终诊断需要医师结合完整临床信息做出。

使用markdown格式，专业且易读。""",
        "parse_mode": "str"
    }
]

# 医学专科配置
MEDICAL_SPECIALTIES = [
    "内科", "外科", "神经科", "心血管科", "消化科", 
    "呼吸科", "内分泌科", "血液科", "肿瘤科", "风湿免疫科"
]

# 系统配置
SYSTEM_CONFIG = {
    "max_execution_time": 300,  # 最大执行时间(秒)
    "retry_count": 2,          # 重试次数
    "log_level": "INFO",       # 日志级别
    "save_intermediate_results": True,  # 保存中间结果
    "medical_disclaimer": True          # 包含医学免责声明
}