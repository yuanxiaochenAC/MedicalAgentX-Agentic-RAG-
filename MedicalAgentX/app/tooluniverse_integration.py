"""
ToolUniverse医学工具集成模块
ToolUniverse Medical Tools Integration Module
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# 添加ToolUniverse到路径
sys.path.append(str(Path(__file__).parent.parent.parent / "ToolUniverse-main" / "src"))

try:
    from tooluniverse.execute_function import ToolUniverse
    TOOLUNIVERSE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ToolUniverse import failed: {e}")
    TOOLUNIVERSE_AVAILABLE = False

class MedicalToolUniverseWrapper:
    """医学ToolUniverse包装器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tool_engine = None
        self._init_tools()
    
    def _init_tools(self):
        """初始化ToolUniverse工具"""
        if not TOOLUNIVERSE_AVAILABLE:
            self.logger.warning("ToolUniverse不可用，将使用模拟工具")
            return
        
        try:
            self.tool_engine = ToolUniverse()
            self.tool_engine.load_tools()
            self.logger.info("ToolUniverse工具引擎初始化成功")
            
            # 显示可用工具
            available_tools = self.tool_engine.get_all_tools_names()
            self.logger.info(f"可用工具数量: {len(available_tools)}")
            
            # 筛选医学相关工具
            self.medical_tools = [tool for tool in available_tools if self._is_medical_tool(tool)]
            self.logger.info(f"医学相关工具: {len(self.medical_tools)}")
            
        except Exception as e:
            self.logger.error(f"初始化ToolUniverse时出错: {str(e)}")
            self.tool_engine = None
    
    def _is_medical_tool(self, tool_name: str) -> bool:
        """判断是否为医学相关工具"""
        medical_keywords = [
            'fda', 'drug', 'disease', 'pharma', 'medical', 'health',
            'opentarget', 'monarch', 'gene', 'protein', 'therapeutic'
        ]
        return any(keyword in tool_name.lower() for keyword in medical_keywords)
    
    def search_drug_information(self, drug_name: str, limit: int = 5) -> Dict[str, Any]:
        """搜索药物信息"""
        if not self.tool_engine:
            return self._mock_drug_search(drug_name)
        
        try:
            # 使用FDA工具搜索药物信息
            result = self.tool_engine.run_one_function({
                "name": "FDA_get_active_ingredient_info_by_drug_name",
                "arguments": {
                    "drug_name": drug_name,
                    "limit": limit,
                    "skip": 0
                }
            })
            
            return {
                "status": "success",
                "tool": "FDA_get_active_ingredient_info_by_drug_name",
                "query": drug_name,
                "results": result
            }
            
        except Exception as e:
            self.logger.error(f"药物信息搜索失败: {str(e)}")
            return self._mock_drug_search(drug_name)
    
    def search_disease_information(self, disease_term: str) -> Dict[str, Any]:
        """搜索疾病信息"""
        if not self.tool_engine:
            return self._mock_disease_search(disease_term)
        
        try:
            # 使用Monarch工具搜索疾病信息
            result = self.tool_engine.run_one_function({
                "name": "MonarchInitiative_get_disease_info_by_name",
                "arguments": {
                    "disease_name": disease_term,
                    "taxon": "NCBITaxon:9606",  # 人类
                    "limit": 5
                }
            })
            
            return {
                "status": "success", 
                "tool": "MonarchInitiative_get_disease_info_by_name",
                "query": disease_term,
                "results": result
            }
            
        except Exception as e:
            self.logger.error(f"疾病信息搜索失败: {str(e)}")
            return self._mock_disease_search(disease_term)
    
    def search_drug_interactions(self, drug_name: str) -> Dict[str, Any]:
        """搜索药物相互作用"""
        if not self.tool_engine:
            return self._mock_drug_interactions(drug_name)
        
        try:
            # 使用OpenTarget工具搜索药物相互作用
            result = self.tool_engine.run_one_function({
                "name": "OpenTarget_get_drug_info_by_name",
                "arguments": {
                    "drug_name": drug_name
                }
            })
            
            return {
                "status": "success",
                "tool": "OpenTarget_get_drug_info_by_name", 
                "query": drug_name,
                "results": result
            }
            
        except Exception as e:
            self.logger.error(f"药物相互作用搜索失败: {str(e)}")
            return self._mock_drug_interactions(drug_name)
    
    def get_treatment_guidelines(self, condition: str) -> Dict[str, Any]:
        """获取治疗指南"""
        if not self.tool_engine:
            return self._mock_treatment_guidelines(condition)
        
        try:
            # 使用多个工具组合查询治疗指南
            results = {}
            
            # 搜索疾病相关信息
            disease_info = self.search_disease_information(condition)
            results["disease_info"] = disease_info
            
            # 如果有相关药物，搜索药物信息
            # 这里可以扩展更复杂的逻辑
            
            return {
                "status": "success",
                "tool": "combined_treatment_search",
                "query": condition,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"治疗指南搜索失败: {str(e)}")
            return self._mock_treatment_guidelines(condition)
    
    def comprehensive_medical_search(self, analysis_text: str, diagnoses: List[str]) -> Dict[str, Any]:
        """综合医学信息搜索"""
        results = {
            "drug_information": {},
            "disease_information": {},
            "treatment_guidelines": {},
            "search_summary": ""
        }
        
        try:
            # 为每个主要诊断搜索信息
            for diagnosis in diagnoses[:3]:  # 限制前3个诊断
                diagnosis = diagnosis.strip()
                if not diagnosis:
                    continue
                
                self.logger.info(f"搜索诊断相关信息: {diagnosis}")
                
                # 搜索疾病信息
                disease_info = self.search_disease_information(diagnosis)
                results["disease_information"][diagnosis] = disease_info
                
                # 搜索治疗指南
                treatment_info = self.get_treatment_guidelines(diagnosis)
                results["treatment_guidelines"][diagnosis] = treatment_info
            
            # 生成搜索摘要
            results["search_summary"] = self._generate_search_summary(results)
            
            return {
                "status": "success",
                "comprehensive_results": results,
                "tools_used": self.medical_tools if self.tool_engine else ["mock_tools"]
            }
            
        except Exception as e:
            self.logger.error(f"综合医学搜索失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "comprehensive_results": results
            }
    
    def _generate_search_summary(self, results: Dict[str, Any]) -> str:
        """生成搜索结果摘要"""
        summary_parts = []
        
        # 疾病信息摘要
        disease_count = len(results.get("disease_information", {}))
        if disease_count > 0:
            summary_parts.append(f"检索了{disease_count}个疾病的相关信息")
        
        # 治疗指南摘要
        treatment_count = len(results.get("treatment_guidelines", {}))
        if treatment_count > 0:
            summary_parts.append(f"获取了{treatment_count}个疾病的治疗指南")
        
        if summary_parts:
            return f"成功{', '.join(summary_parts)}。"
        else:
            return "搜索完成，未找到具体医学信息。"
    
    # 模拟工具方法（当ToolUniverse不可用时使用）
    def _mock_drug_search(self, drug_name: str) -> Dict[str, Any]:
        """模拟药物搜索"""
        return {
            "status": "mock",
            "tool": "mock_drug_search",
            "query": drug_name,
            "results": {
                "message": f"模拟搜索药物 '{drug_name}' 的信息",
                "recommendations": [
                    "请咨询药师了解具体用药信息",
                    "注意药物相互作用和禁忌症",
                    "遵循医嘱使用药物"
                ]
            }
        }
    
    def _mock_disease_search(self, disease_term: str) -> Dict[str, Any]:
        """模拟疾病搜索"""
        return {
            "status": "mock",
            "tool": "mock_disease_search", 
            "query": disease_term,
            "results": {
                "message": f"模拟搜索疾病 '{disease_term}' 的信息",
                "recommendations": [
                    "请参考权威医学指南",
                    "建议进行相关检查确诊",
                    "及时就医获取专业诊疗"
                ]
            }
        }
    
    def _mock_drug_interactions(self, drug_name: str) -> Dict[str, Any]:
        """模拟药物相互作用搜索"""
        return {
            "status": "mock",
            "tool": "mock_drug_interactions",
            "query": drug_name,
            "results": {
                "message": f"模拟搜索药物 '{drug_name}' 的相互作用信息",
                "recommendations": [
                    "请咨询医师或药师了解药物相互作用",
                    "告知医师所有正在使用的药物",
                    "定期监测药物疗效和不良反应"
                ]
            }
        }
    
    def _mock_treatment_guidelines(self, condition: str) -> Dict[str, Any]:
        """模拟治疗指南搜索"""
        return {
            "status": "mock",
            "tool": "mock_treatment_guidelines",
            "query": condition,
            "results": {
                "message": f"模拟搜索 '{condition}' 的治疗指南",
                "recommendations": [
                    "请参考最新的临床诊疗指南",
                    "结合患者具体情况制定治疗方案",
                    "定期评估治疗效果并调整方案"
                ]
            }
        }
    
    def get_tool_status(self) -> Dict[str, Any]:
        """获取工具状态"""
        return {
            "tooluniverse_available": TOOLUNIVERSE_AVAILABLE,
            "tool_engine_initialized": self.tool_engine is not None,
            "medical_tools_count": len(self.medical_tools) if hasattr(self, 'medical_tools') else 0,
            "medical_tools": getattr(self, 'medical_tools', [])
        }

def main():
    """测试ToolUniverse集成"""
    print("🔧 测试ToolUniverse医学工具集成...")
    
    # 创建工具包装器
    tool_wrapper = MedicalToolUniverseWrapper()
    
    # 显示工具状态
    status = tool_wrapper.get_tool_status()
    print("\n📊 工具状态:")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    
    # 测试各种医学工具
    test_cases = [
        ("药物搜索", lambda: tool_wrapper.search_drug_information("阿司匹林")),
        ("疾病搜索", lambda: tool_wrapper.search_disease_information("高血压")),
        ("药物相互作用", lambda: tool_wrapper.search_drug_interactions("华法林")),
        ("治疗指南", lambda: tool_wrapper.get_treatment_guidelines("糖尿病"))
    ]
    
    for test_name, test_func in test_cases:
        print(f"\n🧪 测试: {test_name}")
        try:
            result = test_func()
            print(f"状态: {result.get('status', 'unknown')}")
            print(f"工具: {result.get('tool', 'unknown')}")
            if result.get('results'):
                print(f"结果摘要: {str(result['results'])[:200]}...")
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
    
    # 测试综合搜索
    print(f"\n🔬 测试综合医学搜索...")
    diagnoses = ["高血压", "糖尿病", "心律失常"]
    comprehensive_result = tool_wrapper.comprehensive_medical_search(
        "多种慢性疾病管理", diagnoses
    )
    
    print(f"综合搜索状态: {comprehensive_result.get('status')}")
    print(f"搜索摘要: {comprehensive_result.get('comprehensive_results', {}).get('search_summary', '无摘要')}")

if __name__ == "__main__":
    main()