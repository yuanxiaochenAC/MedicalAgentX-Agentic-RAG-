"""
基于EvoAgentX的医疗智能分析工作流
EvoAgentX-based Medical Intelligence Analysis Workflow
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from evoagentx.core.registry import register_parse_function
from evoagentx.workflow import SequentialWorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager, CustomizeAgent
from evoagentx.core import Message

from evoagentx_medical_config import (
    LLM_CONFIG, MEDICAL_WORKFLOW_TASKS, SYSTEM_CONFIG,
    OUTPUTS_DIR, PROJECT_ROOT
)
from evoagentx_medical_engine import MedicalRAGEngine
from tooluniverse_integration import MedicalToolUniverseWrapper

# 注册自定义解析函数
@register_parse_function
def parse_medical_response(content: str) -> Dict[str, str]:
    """解析医疗分析响应"""
    return {"analysis": content}

class MedicalWorkflowExecutor:
    """基于EvoAgentX的医疗工作流执行器"""
    
    def __init__(self):
        """初始化医疗工作流执行器"""
        self.logger = logging.getLogger(__name__)
        
        # 初始化医疗RAG引擎
        self.medical_rag = MedicalRAGEngine()
        
        # 初始化ToolUniverse医学工具
        self.tool_universe = MedicalToolUniverseWrapper()
        
        # 初始化工作流组件
        self.workflow_graph = None
        self.agent_manager = None
        self.workflow = None
        
        # 初始化工作流
        self._setup_workflow()
        
        self.logger.info("医疗工作流执行器初始化完成")
    
    def _setup_workflow(self):
        """设置EvoAgentX工作流"""
        try:
            # 创建顺序工作流图
            self.workflow_graph = SequentialWorkFlowGraph(
                goal="执行基于AI的医疗病例分析，提供诊断支持和治疗建议",
                tasks=MEDICAL_WORKFLOW_TASKS
            )
            
            # 保存工作流图（可选）
            workflow_file = OUTPUTS_DIR / "medical_workflow_graph.json"
            self.workflow_graph.save_module(str(workflow_file))
            self.logger.info(f"工作流图已保存到: {workflow_file}")
            
            # 创建Agent管理器（不使用外部工具，我们通过RAG引擎处理）
            self.agent_manager = AgentManager(tools=[])
            
            # 从工作流图创建Agent
            self.agent_manager.add_agents_from_workflow(
                self.workflow_graph,
                llm_config=LLM_CONFIG
            )
            
            # 创建LLM实例
            from evoagentx.models import OpenAILLM
            llm = OpenAILLM(LLM_CONFIG)
            
            # 创建工作流实例
            self.workflow = WorkFlow(
                graph=self.workflow_graph,
                agent_manager=self.agent_manager,
                llm=llm
            )
            
            self.logger.info("EvoAgentX工作流设置完成")
            
        except Exception as e:
            self.logger.error(f"设置工作流时发生错误: {str(e)}")
            raise
    
    def ensure_rag_indexed(self, force_reindex: bool = False) -> bool:
        """确保RAG引擎已建立索引"""
        try:
            if not self.medical_rag.is_indexed or force_reindex:
                self.logger.info("正在建立医学文档索引...")
                success = self.medical_rag.index_medical_documents(force_reindex)
                
                if not success:
                    self.logger.error("建立医学文档索引失败")
                    return False
                
                self.logger.info("医学文档索引建立完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"建立索引时发生错误: {str(e)}")
            return False
    
    def execute_medical_analysis(self, symptom_text: str, **kwargs) -> Dict[str, Any]:
        """执行医疗分析工作流"""
        try:
            # 确保RAG索引存在
            if not self.ensure_rag_indexed():
                return {
                    "status": "error",
                    "error": "无法建立医学文档索引",
                    "timestamp": datetime.now().isoformat()
                }
            
            self.logger.info(f"开始执行医疗分析: {symptom_text}")
            
            # 准备输入数据
            workflow_inputs = {
                "symptom_text": symptom_text,
                "top_k": kwargs.get("top_k", 5)
            }
            
            # 在工作流执行前，我们需要先进行RAG检索，然后注入到工作流中
            # 这是因为当前的EvoAgentX工作流不直接支持RAG集成
            
            # 步骤1: 执行RAG检索
            self.logger.info("执行医学文档检索...")
            rag_results = self.medical_rag.search_similar_cases(
                symptom_text, 
                top_k=workflow_inputs["top_k"]
            )
            
            if rag_results["status"] != "success":
                return {
                    "status": "error",
                    "error": f"文档检索失败: {rag_results.get('error', '未知错误')}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 格式化检索结果供工作流使用
            retrieved_docs = self._format_rag_results(rag_results)
            
            # 步骤2: 执行EvoAgentX工作流，并注入RAG结果
            workflow_inputs["retrieved_docs"] = retrieved_docs
            
            self.logger.info("执行EvoAgentX医疗分析工作流...")
            
            # 由于我们需要手动处理RAG集成，我们将分步执行每个Agent
            workflow_results = self._execute_workflow_with_rag(
                workflow_inputs, rag_results
            )
            
            # 保存完整结果
            self._save_workflow_results(workflow_results, symptom_text)
            
            self.logger.info("医疗分析工作流执行完成")
            
            return {
                "status": "success",
                "input": symptom_text,
                "timestamp": datetime.now().isoformat(),
                "rag_results": rag_results,
                "workflow_results": workflow_results,
                "final_report": workflow_results.get("comprehensive_report", ""),
                "executive_summary": workflow_results.get("executive_summary", "")
            }
            
        except Exception as e:
            self.logger.error(f"执行医疗分析时发生错误: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_rag_results(self, rag_results: Dict[str, Any]) -> str:
        """格式化RAG检索结果"""
        if not rag_results.get("results"):
            return "未找到相关医学案例。"
        
        formatted_text = "检索到的相关医学案例:\n\n"
        
        for result in rag_results["results"]:
            formatted_text += f"案例 {result['rank']} (相似度: {result['score']:.4f})\n"
            formatted_text += f"来源: {result['document_title']}\n"
            formatted_text += f"内容: {result['content']}\n"
            formatted_text += "-" * 50 + "\n\n"
        
        return formatted_text
    
    def _execute_workflow_with_rag(
        self, 
        inputs: Dict[str, Any], 
        rag_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行带RAG集成的工作流"""
        
        # 由于需要手动集成RAG，我们直接调用各个Agent
        # 这里我们创建自定义的Agent执行逻辑
        
        results = {}
        
        try:
            # 步骤1: 医学检索Agent (已经通过RAG完成)
            self.logger.info("步骤1: 医学文档检索 - 已完成")
            results["retriever_output"] = {
                "similar_cases": self._format_rag_results(rag_results),
                "retrieval_metadata": json.dumps(rag_results["search_metadata"], ensure_ascii=False)
            }
            
            # 步骤2: 医学推理Agent
            self.logger.info("步骤2: 执行医学推理分析")
            reasoner_agent = self._get_agent_by_name("MedicalreasonerAgent")
            if reasoner_agent:
                reasoner_inputs = {
                    "symptom_text": inputs["symptom_text"],
                    "similar_cases": results["retriever_output"]["similar_cases"],
                    "retrieval_metadata": results["retriever_output"]["retrieval_metadata"]
                }
                
                reasoner_response = reasoner_agent(inputs=reasoner_inputs)
                
                # 解析响应内容
                response_content = reasoner_response.content.content
                results["reasoner_output"] = {
                    "medical_analysis": response_content,
                    "primary_diagnoses": self._extract_diagnoses(response_content),
                    "recommended_tests": self._extract_tests(response_content)
                }
            
            # 步骤3: 医学工具咨询Agent (使用真实的ToolUniverse)
            self.logger.info("步骤3: 执行医学工具咨询")
            
            # 首先使用ToolUniverse进行真实的医学工具查询
            diagnoses_list = self._parse_diagnoses_list(results["reasoner_output"]["primary_diagnoses"])
            tooluniverse_results = self.tool_universe.comprehensive_medical_search(
                results["reasoner_output"]["medical_analysis"],
                diagnoses_list
            )
            
            # 然后调用LLM Agent进行结果整理
            tools_agent = self._get_agent_by_name("MedicaltoolsconsultantAgent")
            if tools_agent:
                tools_inputs = {
                    "medical_analysis": results["reasoner_output"]["medical_analysis"],
                    "primary_diagnoses": results["reasoner_output"]["primary_diagnoses"]
                }
                
                tools_response = tools_agent(inputs=tools_inputs)
                tools_content = tools_response.content.content
                
                # 合并ToolUniverse结果和LLM分析
                results["tools_output"] = {
                    "tool_consultation": tools_content,
                    "additional_guidance": self._extract_guidance(tools_content),
                    "tooluniverse_results": tooluniverse_results,
                    "real_medical_tools": self.tool_universe.get_tool_status()
                }
            
            # 步骤4: 医学报告生成Agent
            self.logger.info("步骤4: 生成综合医学报告")
            report_agent = self._get_agent_by_name("MedicalreportgeneratorAgent")
            if report_agent:
                report_inputs = {
                    "symptom_text": inputs["symptom_text"],
                    "similar_cases": results["retriever_output"]["similar_cases"],
                    "medical_analysis": results["reasoner_output"]["medical_analysis"],
                    "primary_diagnoses": results["reasoner_output"]["primary_diagnoses"],
                    "recommended_tests": results["reasoner_output"]["recommended_tests"],
                    "tool_consultation": results["tools_output"]["tool_consultation"],
                    "additional_guidance": results["tools_output"]["additional_guidance"]
                }
                
                report_response = report_agent(inputs=report_inputs)
                report_content = report_response.content.content
                
                results["report_output"] = {
                    "comprehensive_report": report_content,
                    "executive_summary": self._extract_summary(report_content)
                }
                
                # 添加最终结果到根级别
                results["comprehensive_report"] = report_content
                results["executive_summary"] = self._extract_summary(report_content)
            
            return results
            
        except Exception as e:
            self.logger.error(f"执行工作流步骤时发生错误: {str(e)}")
            results["error"] = str(e)
            return results
    
    def _get_agent_by_name(self, agent_name: str) -> CustomizeAgent:
        """通过名称获取Agent"""
        if self.agent_manager and hasattr(self.agent_manager, 'agents'):
            for agent in self.agent_manager.agents:
                if agent.name == agent_name:
                    return agent
        return None
    
    def _extract_diagnoses(self, content: str) -> str:
        """从分析内容中提取诊断信息"""
        # 简单的诊断提取逻辑
        lines = content.split('\n')
        diagnoses = []
        
        in_diagnosis_section = False
        for line in lines:
            line = line.strip()
            if "可能病因" in line or "诊断" in line:
                in_diagnosis_section = True
                continue
            elif in_diagnosis_section and line and not line.startswith('#'):
                if any(keyword in line for keyword in ['检查', '治疗', '风险']):
                    break
                diagnoses.append(line)
        
        return "\n".join(diagnoses) if diagnoses else "需要进一步分析确定诊断方向"
    
    def _extract_tests(self, content: str) -> str:
        """从分析内容中提取检查建议"""
        lines = content.split('\n')
        tests = []
        
        in_tests_section = False
        for line in lines:
            line = line.strip()
            if "检查建议" in line or "检查" in line:
                in_tests_section = True
                continue
            elif in_tests_section and line and not line.startswith('#'):
                if any(keyword in line for keyword in ['治疗', '风险', '随访']):
                    break
                tests.append(line)
        
        return "\n".join(tests) if tests else "建议常规实验室检查和影像学检查"
    
    def _extract_guidance(self, content: str) -> str:
        """从工具咨询内容中提取指导建议"""
        # 返回前500字符作为核心指导
        return content[:500] + "..." if len(content) > 500 else content
    
    def _parse_diagnoses_list(self, diagnoses_text: str) -> List[str]:
        """从诊断文本中解析出诊断列表"""
        diagnoses = []
        lines = diagnoses_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # 移除序号和特殊字符
                clean_line = line.lstrip('0123456789.-、 ')
                if clean_line and len(clean_line) > 2:
                    diagnoses.append(clean_line)
        
        return diagnoses[:5]  # 限制最多5个诊断
    
    def _extract_summary(self, content: str) -> str:
        """从报告内容中提取执行摘要"""
        lines = content.split('\n')
        summary_lines = []
        
        in_summary_section = False
        for line in lines:
            line = line.strip()
            if "执行摘要" in line:
                in_summary_section = True
                continue
            elif in_summary_section:
                if line.startswith('#') and "执行摘要" not in line:
                    break
                if line and not line.startswith('#'):
                    summary_lines.append(line)
        
        return "\n".join(summary_lines) if summary_lines else "请查看完整报告获取详细信息"
    
    def _save_workflow_results(self, results: Dict[str, Any], query: str):
        """保存工作流执行结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evoagentx_medical_analysis_{timestamp}.json"
            filepath = OUTPUTS_DIR / "results" / filename
            
            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存结果
            save_data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "executor_type": "EvoAgentX",
                "workflow_config": {
                    "goal": self.workflow_graph.goal if self.workflow_graph else "",
                    "tasks_count": len(MEDICAL_WORKFLOW_TASKS),
                    "llm_model": LLM_CONFIG.model
                },
                "results": results
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"工作流结果已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存工作流结果时出错: {str(e)}")
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
        return {
            "executor_type": "EvoAgentX",
            "workflow_goal": self.workflow_graph.goal if self.workflow_graph else "",
            "tasks_count": len(MEDICAL_WORKFLOW_TASKS),
            "agents_count": len(self.agent_manager.agents) if self.agent_manager else 0,
            "llm_config": {
                "model": LLM_CONFIG.model,
                "temperature": LLM_CONFIG.temperature,
                "max_tokens": LLM_CONFIG.max_tokens
            },
            "rag_engine_info": self.medical_rag.get_corpus_info(),
            "tooluniverse_status": self.tool_universe.get_tool_status(),
            "system_config": SYSTEM_CONFIG
        }

def main():
    """测试EvoAgentX医疗工作流"""
    print("🏥 初始化EvoAgentX医疗工作流执行器...")
    
    try:
        # 创建执行器
        executor = MedicalWorkflowExecutor()
        
        # 显示工作流信息
        workflow_info = executor.get_workflow_info()
        print("\n📊 工作流信息:")
        print(json.dumps(workflow_info, ensure_ascii=False, indent=2))
        
        # 测试用例
        test_cases = [
            "患者男性，58岁，近期出现体重下降，皮肤黄疸，上腹部胀痛，ALT和胆红素升高",
            "女性患者，35岁，反复头痛伴恶心呕吐，视物模糊，血压180/110mmHg"
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"🧪 测试案例 {i}: {case}")
            print('='*60)
            
            # 执行分析
            results = executor.execute_medical_analysis(case)
            
            if results["status"] == "success":
                print("✅ 分析成功完成")
                print(f"\n📋 执行摘要:")
                print(results.get("executive_summary", "无摘要"))
                
                print(f"\n🩺 完整报告:")
                print(results.get("final_report", "无报告"))
                
            else:
                print(f"❌ 分析失败: {results.get('error', '未知错误')}")
        
        print(f"\n🎉 EvoAgentX医疗工作流测试完成!")
        
    except Exception as e:
        print(f"❌ 执行过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()