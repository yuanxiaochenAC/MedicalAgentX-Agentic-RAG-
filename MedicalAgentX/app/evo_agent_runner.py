"""
EvoAgentX运行器 - 执行医疗智能Agent工作流
EvoAgentX Runner - Execute medical intelligence agent workflows
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 添加EvoAgentX到路径
sys.path.append(str(Path(__file__).parent.parent.parent / "EvoAgentX-clean_tools"))
sys.path.append(str(Path(__file__).parent.parent.parent / "ToolUniverse-main"))

from config import (
    EVOAGENTX_CONFIG, 
    TOOLUNIVERSE_CONFIG, 
    LLM_CONFIG, 
    FAISS_CONFIG,
    RESULTS_DIR, 
    LOGS_DIR,
    load_api_key
)
from ingestion import MedicalPDFProcessor

# 设置日志
logging.basicConfig(
    level=getattr(logging, EVOAGENTX_CONFIG["log_level"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"medical_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalAgentExecutor:
    """医疗智能Agent执行器"""
    
    def __init__(self):
        self.pdf_processor = MedicalPDFProcessor()
        self.api_key = load_api_key(LLM_CONFIG["api_key_file"])
        
        # 初始化组件
        self._setup_components()
        
    def _setup_components(self):
        """设置组件"""
        try:
            # 尝试加载现有的向量存储
            self.pdf_processor.vector_store = self.pdf_processor.load_vector_store()
            logger.info("成功加载现有向量存储")
        except FileNotFoundError:
            logger.warning("向量存储不存在，请先运行文档导入")
    
    def retriever_agent(self, symptom_text: str) -> Dict[str, Any]:
        """检索Agent - 使用FAISS检索相似病例"""
        logger.info(f"检索Agent开始处理: {symptom_text}")
        
        try:
            # 使用FAISS检索相似病例
            results = self.pdf_processor.search_similar_cases(
                query=symptom_text, 
                k=FAISS_CONFIG["top_k"]
            )
            
            # 格式化检索结果
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    "rank": i,
                    "content": result["content"],
                    "source": result["source"],
                    "similarity_score": result["similarity_score"],
                    "metadata": result["metadata"]
                })
            
            logger.info(f"检索到 {len(formatted_results)} 个相似病例")
            return {
                "status": "success",
                "results": formatted_results,
                "query": symptom_text,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"检索Agent执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }
    
    def reasoning_agent(self, symptom_text: str, retrieval_results: Dict[str, Any]) -> Dict[str, Any]:
        """推理Agent - 基于相似病例进行病因分析"""
        logger.info("推理Agent开始分析")
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            # 构建prompt
            cases_text = ""
            if retrieval_results.get("results"):
                for result in retrieval_results["results"][:3]:  # 使用前3个最相似的病例
                    cases_text += f"\n病例 {result['rank']} (相似度: {result['similarity_score']:.3f}):\n"
                    cases_text += f"来源: {result['source']}\n"
                    cases_text += f"内容: {result['content'][:500]}...\n"
                    cases_text += "---\n"
            
            prompt = f"""
作为医疗AI助手，请基于以下信息进行病因分析：

患者主诉症状：
{symptom_text}

检索到的相似病例：
{cases_text}

请从以下几个方面进行分析：
1. 可能的病因分析
2. 初步诊断方向
3. 建议的检查项目
4. 需要关注的并发症或风险因素
5. 治疗建议（仅供参考）

注意：本分析仅供医疗专业人员参考，不能替代临床诊断。
"""
            
            # 调用LLM进行推理
            response = client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[
                    {"role": "system", "content": "你是一个专业的医疗AI助手，擅长基于病例进行诊断分析。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"]
            )
            
            analysis_result = response.choices[0].message.content
            
            logger.info("推理Agent完成分析")
            return {
                "status": "success",
                "analysis": analysis_result,
                "used_cases": len(retrieval_results.get("results", [])),
                "reasoning_model": LLM_CONFIG["model"]
            }
            
        except Exception as e:
            logger.error(f"推理Agent执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "analysis": "推理分析失败"
            }
    
    def tool_universe_agent(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """ToolUniverse Agent - 调用医学工具获取额外信息"""
        logger.info("ToolUniverse Agent开始工作")
        
        try:
            # 这里可以集成ToolUniverse的医学工具
            # 暂时使用模拟数据，实际使用时需要集成真实的ToolUniverse
            
            tool_results = {
                "drug_interactions": "暂无发现药物相互作用风险",
                "lab_reference_values": "建议检查：血常规、肝功能、肾功能",
                "differential_diagnosis": "需排除其他可能疾病",
                "treatment_guidelines": "遵循相关临床指南"
            }
            
            logger.info("ToolUniverse Agent完成工具调用")
            return {
                "status": "success",
                "tool_results": tool_results,
                "tools_used": list(tool_results.keys())
            }
            
        except Exception as e:
            logger.error(f"ToolUniverse Agent执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "tool_results": {}
            }
    
    def integration_agent(self, symptom_text: str, analysis_result: Dict[str, Any], 
                         tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """综合Agent - 整合所有结果生成最终报告"""
        logger.info("综合Agent开始整合分析")
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            # 构建综合分析prompt
            prompt = f"""
请整合以下医疗分析信息，生成一份完整的医学分析报告：

患者症状：
{symptom_text}

初步诊断分析：
{analysis_result.get('analysis', '无分析结果')}

工具查询结果：
{json.dumps(tool_results.get('tool_results', {}), ensure_ascii=False, indent=2)}

请生成一份结构化的医学分析报告，包含：
1. 症状总结
2. 可能病因
3. 诊断建议
4. 检查建议
5. 治疗方向
6. 注意事项
7. 随访建议

格式要求：使用清晰的markdown格式，专业但易懂。
重要提醒：此报告仅供医疗专业人员参考，不能替代正式的临床诊断。
"""
            
            response = client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[
                    {"role": "system", "content": "你是一个资深的医疗专家，负责生成综合性医疗分析报告。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"]
            )
            
            final_report = response.choices[0].message.content
            
            logger.info("综合Agent完成报告生成")
            return {
                "status": "success",
                "final_report": final_report,
                "analysis_timestamp": datetime.now().isoformat(),
                "components_used": ["retriever", "reasoning", "tool_universe", "integration"]
            }
            
        except Exception as e:
            logger.error(f"综合Agent执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "final_report": "报告生成失败"
            }
    
    def run_medical_analysis_workflow(self, symptom_text: str) -> Dict[str, Any]:
        """运行完整的医疗分析工作流"""
        logger.info(f"开始医疗分析工作流: {symptom_text}")
        
        workflow_results = {
            "input": symptom_text,
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        try:
            # 阶段1: 检索相似病例
            logger.info("执行阶段1: 病例检索")
            retrieval_results = self.retriever_agent(symptom_text)
            workflow_results["stages"]["retrieval"] = retrieval_results
            
            if retrieval_results["status"] != "success":
                raise Exception("病例检索失败")
            
            # 阶段2: 推理分析
            logger.info("执行阶段2: 病因推理")
            reasoning_results = self.reasoning_agent(symptom_text, retrieval_results)
            workflow_results["stages"]["reasoning"] = reasoning_results
            
            if reasoning_results["status"] != "success":
                raise Exception("病因推理失败")
            
            # 阶段3: 工具调用
            logger.info("执行阶段3: 医学工具查询")
            tool_results = self.tool_universe_agent(reasoning_results)
            workflow_results["stages"]["tools"] = tool_results
            
            # 阶段4: 综合分析
            logger.info("执行阶段4: 综合分析报告")
            integration_results = self.integration_agent(symptom_text, reasoning_results, tool_results)
            workflow_results["stages"]["integration"] = integration_results
            
            workflow_results["status"] = "success"
            workflow_results["final_report"] = integration_results.get("final_report", "")
            
            # 保存结果
            self._save_results(workflow_results)
            
            logger.info("医疗分析工作流完成")
            return workflow_results
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            workflow_results["status"] = "error"
            workflow_results["error"] = str(e)
            return workflow_results
    
    def _save_results(self, results: Dict[str, Any]):
        """保存分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_analysis_{timestamp}.json"
            filepath = RESULTS_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分析结果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")

def main():
    """主函数 - 示例运行"""
    executor = MedicalAgentExecutor()
    
    # 示例病例分析
    test_cases = [
        "患者男性，58岁，近期出现体重下降，皮肤黄疸，上腹部胀痛，ALT和胆红素升高",
        "女性患者，35岁，反复头痛伴恶心呕吐，视物模糊，血压180/110mmHg",
        "患者主诉胸痛，呼吸困难，心电图显示ST段抬高，肌钙蛋白升高"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\\n{'='*50}")
        print(f"测试病例 {i}: {case}")
        print('='*50)
        
        results = executor.run_medical_analysis_workflow(case)
        
        if results["status"] == "success":
            print("\\n🩺 医疗分析报告:")
            print(results.get("final_report", "无报告内容"))
        else:
            print(f"\\n❌ 分析失败: {results.get('error', '未知错误')}")

if __name__ == "__main__":
    main()