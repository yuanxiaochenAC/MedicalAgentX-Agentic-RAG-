"""
医疗智能Agent系统配置文件
Configuration file for Medical Intelligence Agent System
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "medical_cases"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
RESULTS_DIR = OUTPUTS_DIR / "results"

# FAISS数据库配置
FAISS_CONFIG = {
    "corpus_id": "medical_case_reports",
    "index_path": str(PROJECT_ROOT / "data" / "faiss_index"),
    "embedding_model": "text-embedding-ada-002",  # OpenAI embedding模型
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5  # 默认检索前5个相似文档
}

# LLM配置
LLM_CONFIG = {
    "model": "gpt-4",  # 或者其他模型
    "temperature": 0.3,  # 较低的温度确保医学推理的准确性
    "max_tokens": 2000,
    "api_key_file": str(Path(__file__).parent.parent.parent / "openai_api_key.txt")
}

# EvoAgentX配置
EVOAGENTX_CONFIG = {
    "agents_dir": str(PROJECT_ROOT / "agents"),
    "workflows_dir": str(PROJECT_ROOT / "workflows"),
    "log_level": "INFO"
}

# ToolUniverse配置
TOOLUNIVERSE_CONFIG = {
    "base_path": str(Path(__file__).parent.parent.parent / "ToolUniverse-main"),
    "enabled_tools": [
        "medical_drugs",
        "medical_symptoms", 
        "medical_diagnostics",
        "medical_procedures"
    ]
}

def load_api_key(key_file_path: str) -> str:
    """加载API密钥"""
    try:
        with open(key_file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key文件未找到: {key_file_path}")

def ensure_directories():
    """确保所有必要的目录存在"""
    for directory in [DATA_DIR, OUTPUTS_DIR, LOGS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# 医学领域特定配置
MEDICAL_CONFIG = {
    "specialties": [
        "内科", "外科", "神经科", "心血管科", "肿瘤科", 
        "消化科", "呼吸科", "内分泌科", "血液科", "风湿免疫科"
    ],
    "symptom_keywords": [
        "疼痛", "发热", "咳嗽", "呼吸困难", "腹痛", "头痛", 
        "恶心", "呕吐", "腹泻", "便血", "黄疸", "水肿"
    ],
    "diagnosis_confidence_threshold": 0.7  # 诊断置信度阈值
}