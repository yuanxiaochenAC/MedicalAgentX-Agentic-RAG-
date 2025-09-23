#!/usr/bin/env python3
"""
MedicalAgentX系统测试脚本
System Test Script for MedicalAgentX
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加app目录到路径
sys.path.append(str(Path(__file__).parent / "app"))

from config import ensure_directories, RESULTS_DIR, LOGS_DIR
from ingestion import MedicalPDFProcessor
from evo_agent_runner import MedicalAgentExecutor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalAgentXTester:
    """MedicalAgentX系统测试器"""
    
    def __init__(self):
        ensure_directories()
        self.test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "tests": {}
        }
    
    def test_pdf_ingestion(self) -> bool:
        """测试PDF文档导入功能"""
        logger.info("🧪 测试PDF文档导入功能...")
        
        try:
            processor = MedicalPDFProcessor()
            
            # 检查PDF文件
            from config import DATA_DIR
            pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
            
            if not pdf_files:
                logger.warning("⚠️ 未发现PDF文件，创建测试数据...")
                # 这里可以创建测试PDF或跳过
                self.test_results["tests"]["pdf_ingestion"] = {
                    "status": "skipped",
                    "reason": "no_pdf_files",
                    "message": "未发现PDF文件进行测试"
                }
                return False
            
            logger.info(f"发现 {len(pdf_files)} 个PDF文件")
            
            # 处理PDF文档
            documents = processor.process_pdf_files()
            
            if documents:
                logger.info(f"✅ 成功处理 {len(documents)} 个文档块")
                
                # 创建向量存储
                vector_store = processor.create_vector_store(documents)
                
                # 测试搜索功能
                test_query = "头痛症状"
                results = processor.search_similar_cases(test_query, k=3)
                
                self.test_results["tests"]["pdf_ingestion"] = {
                    "status": "success",
                    "documents_processed": len(documents),
                    "pdf_files_count": len(pdf_files),
                    "search_test": {
                        "query": test_query,
                        "results_count": len(results),
                        "top_similarity": results[0]["similarity_score"] if results else 0
                    }
                }
                
                logger.info("✅ PDF导入测试通过")
                return True
            else:
                raise Exception("未能处理任何PDF文档")
                
        except Exception as e:
            logger.error(f"❌ PDF导入测试失败: {str(e)}")
            self.test_results["tests"]["pdf_ingestion"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_agent_execution(self) -> bool:
        """测试Agent执行功能"""
        logger.info("🧪 测试Agent执行功能...")
        
        try:
            executor = MedicalAgentExecutor()
            
            # 测试用例
            test_cases = [
                {
                    "name": "基础症状分析",
                    "symptoms": "患者主诉头痛，伴有恶心呕吐",
                    "expected_components": ["retrieval", "reasoning", "integration"]
                },
                {
                    "name": "复杂病例分析", 
                    "symptoms": "男性患者，58岁，体重下降，黄疸，上腹部疼痛，ALT升高",
                    "expected_components": ["retrieval", "reasoning", "tools", "integration"]
                }
            ]
            
            test_results = []
            
            for test_case in test_cases:
                logger.info(f"测试案例: {test_case['name']}")
                
                try:
                    # 执行工作流
                    result = executor.run_medical_analysis_workflow(test_case["symptoms"])
                    
                    # 验证结果
                    case_result = {
                        "case_name": test_case["name"],
                        "status": result.get("status", "unknown"),
                        "stages_completed": list(result.get("stages", {}).keys()),
                        "has_final_report": bool(result.get("final_report", "")),
                        "execution_time": result.get("timestamp", "")
                    }
                    
                    if result.get("status") == "success":
                        logger.info(f"✅ 测试案例 '{test_case['name']}' 通过")
                        case_result["success"] = True
                    else:
                        logger.warning(f"⚠️ 测试案例 '{test_case['name']}' 部分失败")
                        case_result["success"] = False
                        case_result["error"] = result.get("error", "unknown error")
                    
                    test_results.append(case_result)
                    
                except Exception as e:
                    logger.error(f"❌ 测试案例 '{test_case['name']}' 失败: {str(e)}")
                    test_results.append({
                        "case_name": test_case["name"],
                        "status": "failed",
                        "success": False,
                        "error": str(e)
                    })
            
            # 统计结果
            successful_cases = sum(1 for r in test_results if r.get("success", False))
            total_cases = len(test_results)
            
            self.test_results["tests"]["agent_execution"] = {
                "status": "success" if successful_cases > 0 else "failed",
                "total_cases": total_cases,
                "successful_cases": successful_cases,
                "success_rate": successful_cases / total_cases if total_cases > 0 else 0,
                "test_cases": test_results
            }
            
            logger.info(f"✅ Agent执行测试完成: {successful_cases}/{total_cases} 案例成功")
            return successful_cases > 0
            
        except Exception as e:
            logger.error(f"❌ Agent执行测试失败: {str(e)}")
            self.test_results["tests"]["agent_execution"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_configuration(self) -> bool:
        """测试系统配置"""
        logger.info("🧪 测试系统配置...")
        
        try:
            from config import (
                FAISS_CONFIG, LLM_CONFIG, EVOAGENTX_CONFIG, 
                TOOLUNIVERSE_CONFIG, load_api_key
            )
            
            config_checks = {
                "faiss_config": bool(FAISS_CONFIG.get("corpus_id")),
                "llm_config": bool(LLM_CONFIG.get("model")),
                "evoagentx_config": bool(EVOAGENTX_CONFIG.get("agents_dir")),
                "tooluniverse_config": bool(TOOLUNIVERSE_CONFIG.get("base_path"))
            }
            
            # 测试API密钥加载
            try:
                api_key = load_api_key(LLM_CONFIG["api_key_file"])
                config_checks["api_key"] = bool(api_key and len(api_key) > 10)
            except Exception as e:
                logger.warning(f"API密钥测试失败: {str(e)}")
                config_checks["api_key"] = False
            
            # 检查目录结构
            required_dirs = ["agents", "workflows", "data", "outputs"]
            for dir_name in required_dirs:
                dir_path = Path(__file__).parent / dir_name
                config_checks[f"dir_{dir_name}"] = dir_path.exists()
            
            self.test_results["tests"]["configuration"] = {
                "status": "success" if all(config_checks.values()) else "partial",
                "checks": config_checks,
                "failed_checks": [k for k, v in config_checks.items() if not v]
            }
            
            failed_count = sum(1 for v in config_checks.values() if not v)
            if failed_count == 0:
                logger.info("✅ 配置测试全部通过")
                return True
            else:
                logger.warning(f"⚠️ 配置测试部分失败: {failed_count} 项检查未通过")
                return False
                
        except Exception as e:
            logger.error(f"❌ 配置测试失败: {str(e)}")
            self.test_results["tests"]["configuration"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def run_full_test_suite(self) -> dict:
        """运行完整测试套件"""
        logger.info("🚀 开始MedicalAgentX系统测试...")
        
        # 运行各项测试
        tests = [
            ("配置测试", self.test_configuration),
            ("PDF导入测试", self.test_pdf_ingestion),
            ("Agent执行测试", self.test_agent_execution)
        ]
        
        overall_success = True
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"执行: {test_name}")
            logger.info(f"{'='*50}")
            
            success = test_func()
            overall_success = overall_success and success
            
            if success:
                logger.info(f"✅ {test_name} 通过")
            else:
                logger.error(f"❌ {test_name} 失败")
        
        # 生成测试总结
        self.test_results["overall_status"] = "success" if overall_success else "failed"
        self.test_results["summary"] = {
            "total_tests": len(tests),
            "passed_tests": sum(1 for _, test_func in tests if self.test_results["tests"].get(test_func.__name__.replace("test_", ""), {}).get("status") == "success"),
            "overall_success": overall_success
        }
        
        # 保存测试结果
        self._save_test_results()
        
        logger.info(f"\n{'='*50}")
        logger.info("🏁 测试完成")
        logger.info(f"总体状态: {'✅ 成功' if overall_success else '❌ 失败'}")
        logger.info(f"{'='*50}")
        
        return self.test_results
    
    def _save_test_results(self):
        """保存测试结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_test_results_{timestamp}.json"
            filepath = RESULTS_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 测试结果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存测试结果失败: {str(e)}")

def main():
    """主函数"""
    print("🏥 MedicalAgentX 系统测试")
    print("=" * 50)
    
    tester = MedicalAgentXTester()
    results = tester.run_full_test_suite()
    
    # 输出结果摘要
    print("\n📊 测试结果摘要:")
    print(f"总体状态: {results['overall_status']}")
    print(f"测试项目: {results['summary']['total_tests']}")
    print(f"通过项目: {results['summary']['passed_tests']}")
    
    # 输出详细结果
    print("\n📋 详细结果:")
    for test_name, test_result in results["tests"].items():
        status_emoji = "✅" if test_result["status"] == "success" else ("⚠️" if test_result["status"] == "partial" else "❌")
        print(f"  {status_emoji} {test_name}: {test_result['status']}")
        if test_result.get("error"):
            print(f"    错误: {test_result['error']}")
    
    return results

if __name__ == "__main__":
    main()