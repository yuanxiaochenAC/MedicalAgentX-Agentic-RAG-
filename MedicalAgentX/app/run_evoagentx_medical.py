#!/usr/bin/env python3
"""
EvoAgentX医疗智能系统主执行程序
EvoAgentX Medical Intelligence System Main Runner
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加EvoAgentX到路径
sys.path.append(str(Path(__file__).parent.parent.parent / "EvoAgentX-clean_tools"))

from evoagentx_medical_workflow import MedicalWorkflowExecutor
from evoagentx_medical_config import OUTPUTS_DIR, SYSTEM_CONFIG

def setup_logging(log_level: str = "INFO"):
    """设置日志系统"""
    log_dir = OUTPUTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evoagentx_medical_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def interactive_mode(executor: MedicalWorkflowExecutor):
    """交互模式 - 允许用户输入症状进行分析"""
    print("\n🏥 EvoAgentX医疗智能分析系统 - 交互模式")
    print("=" * 60)
    print("请输入患者症状描述，系统将为您提供AI辅助的医学分析。")
    print("输入 'quit' 或 'exit' 退出程序。")
    print("=" * 60)
    
    while True:
        try:
            # 获取用户输入
            symptom_input = input("\n🔍 请输入患者症状描述: ").strip()
            
            if not symptom_input:
                print("⚠️ 请输入有效的症状描述")
                continue
            
            if symptom_input.lower() in ['quit', 'exit', '退出']:
                print("👋 谢谢使用！再见！")
                break
            
            print(f"\n🚀 正在分析: {symptom_input}")
            print("⏳ 请稍等，系统正在进行智能分析...")
            
            # 执行分析
            results = executor.execute_medical_analysis(symptom_input)
            
            # 显示结果
            print("\n" + "="*60)
            if results["status"] == "success":
                print("✅ 分析完成！")
                
                # 显示执行摘要
                if results.get("executive_summary"):
                    print(f"\n📋 执行摘要:")
                    print("-" * 40)
                    print(results["executive_summary"])
                
                # 显示完整报告
                if results.get("final_report"):
                    print(f"\n🩺 详细医学分析报告:")
                    print("-" * 40)
                    print(results["final_report"])
                
                # 显示检索信息
                rag_results = results.get("rag_results", {})
                if rag_results.get("total_results", 0) > 0:
                    print(f"\n📚 参考了 {rag_results['total_results']} 个相似医学案例")
                
                print(f"\n⏰ 分析完成时间: {results['timestamp']}")
                
            else:
                print(f"❌ 分析失败: {results.get('error', '未知错误')}")
            
            print("="*60)
            
        except KeyboardInterrupt:
            print(f"\n\n👋 用户中断，程序退出")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")

def batch_mode(executor: MedicalWorkflowExecutor, cases_file: str):
    """批处理模式 - 从文件读取多个病例进行分析"""
    print(f"\n🏥 EvoAgentX医疗智能分析系统 - 批处理模式")
    print(f"📁 正在处理文件: {cases_file}")
    
    try:
        with open(cases_file, 'r', encoding='utf-8') as f:
            cases = [line.strip() for line in f if line.strip()]
        
        if not cases:
            print("❌ 文件中没有找到有效的病例描述")
            return
        
        print(f"📋 发现 {len(cases)} 个病例，开始批量分析...")
        
        results_summary = []
        
        for i, case in enumerate(cases, 1):
            print(f"\n{'='*60}")
            print(f"🧪 正在分析病例 {i}/{len(cases)}")
            print(f"📝 症状: {case}")
            print(f"{'='*60}")
            
            # 执行分析
            result = executor.execute_medical_analysis(case)
            
            if result["status"] == "success":
                print("✅ 分析成功")
                results_summary.append({
                    "case_number": i,
                    "symptoms": case,
                    "status": "success",
                    "timestamp": result["timestamp"]
                })
                
                # 显示简要结果
                if result.get("executive_summary"):
                    print(f"📋 摘要: {result['executive_summary'][:200]}...")
                
            else:
                print(f"❌ 分析失败: {result.get('error', '未知错误')}")
                results_summary.append({
                    "case_number": i,
                    "symptoms": case,
                    "status": "failed",
                    "error": result.get('error', '未知错误')
                })
        
        # 显示批处理摘要
        print(f"\n🎉 批处理完成！")
        print(f"📊 处理摘要:")
        successful = sum(1 for r in results_summary if r["status"] == "success")
        failed = len(results_summary) - successful
        print(f"  ✅ 成功: {successful}/{len(results_summary)}")
        print(f"  ❌ 失败: {failed}/{len(results_summary)}")
        
        # 保存批处理结果摘要
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = OUTPUTS_DIR / "results" / f"batch_summary_{timestamp}.json"
        
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        
        print(f"📁 批处理摘要已保存到: {summary_file}")
        
    except FileNotFoundError:
        print(f"❌ 文件未找到: {cases_file}")
    except Exception as e:
        print(f"❌ 批处理过程中发生错误: {str(e)}")

def single_query_mode(executor: MedicalWorkflowExecutor, query: str):
    """单次查询模式 - 分析指定的症状描述"""
    print(f"\n🏥 EvoAgentX医疗智能分析系统 - 单次查询模式")
    print(f"🔍 正在分析症状: {query}")
    print("⏳ 请稍等，系统正在进行智能分析...")
    
    try:
        # 执行分析
        result = executor.execute_medical_analysis(query)
        
        # 显示结果
        print("\n" + "="*60)
        if result["status"] == "success":
            print("✅ 分析完成！")
            
            # 显示执行摘要
            if result.get("executive_summary"):
                print(f"\n📋 执行摘要:")
                print("-" * 40)
                print(result["executive_summary"])
            
            # 显示完整报告
            if result.get("final_report"):
                print(f"\n🩺 详细医学分析报告:")
                print("-" * 40)
                print(result["final_report"])
            
            # 显示检索信息
            rag_results = result.get("rag_results", {})
            if rag_results.get("total_results", 0) > 0:
                print(f"\n📚 参考了 {rag_results['total_results']} 个相似医学案例")
            
            print(f"\n⏰ 分析完成时间: {result['timestamp']}")
            
        else:
            print(f"❌ 分析失败: {result.get('error', '未知错误')}")
        
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")

def demo_mode(executor: MedicalWorkflowExecutor):
    """演示模式 - 运行预设的演示案例"""
    print("\n🏥 EvoAgentX医疗智能分析系统 - 演示模式")
    
    demo_cases = [
        {
            "name": "肝胆疾病案例",
            "symptoms": "患者男性，58岁，近期出现体重下降，皮肤黄疸，上腹部胀痛，ALT和胆红素升高",
            "description": "典型的肝胆系统疾病表现"
        },
        {
            "name": "高血压危象案例", 
            "symptoms": "女性患者，35岁，反复头痛伴恶心呕吐，视物模糊，血压180/110mmHg",
            "description": "可能的高血压急症或颅内病变"
        },
        {
            "name": "急性心肌梗死案例",
            "symptoms": "患者主诉胸痛，呼吸困难，心电图显示ST段抬高，肌钙蛋白升高",
            "description": "典型的急性ST段抬高型心肌梗死"
        }
    ]
    
    print(f"🎭 将演示 {len(demo_cases)} 个医学案例的AI分析过程")
    
    for i, case in enumerate(demo_cases, 1):
        print(f"\n{'='*70}")
        print(f"🎬 演示案例 {i}: {case['name']}")
        print(f"📝 病例描述: {case['description']}")
        print(f"🔍 症状描述: {case['symptoms']}")
        print(f"{'='*70}")
        
        input("\n⏸️  按回车键开始分析...")
        
        # 执行分析
        result = executor.execute_medical_analysis(case['symptoms'])
        
        if result["status"] == "success":
            print("✅ 分析完成！")
            
            # 显示关键结果
            if result.get("executive_summary"):
                print(f"\n📋 分析摘要:")
                print("-" * 50)
                print(result["executive_summary"])
            
            if result.get("final_report"):
                print(f"\n🩺 完整分析报告:")
                print("-" * 50)
                # 显示报告的前1000字符
                report = result["final_report"]
                if len(report) > 1000:
                    print(report[:1000] + "\n\n[报告已截断，完整内容请查看保存的文件]")
                else:
                    print(report)
            
            rag_info = result.get("rag_results", {})
            if rag_info.get("total_results", 0) > 0:
                print(f"\n📚 参考医学案例: {rag_info['total_results']} 个")
        
        else:
            print(f"❌ 分析失败: {result.get('error', '未知错误')}")
        
        if i < len(demo_cases):
            input(f"\n⏸️  按回车键继续下一个演示案例...")
    
    print(f"\n🎉 演示完成！所有分析结果已保存到 {OUTPUTS_DIR}/results/ 目录")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="EvoAgentX医疗智能分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_evoagentx_medical.py                                           # 交互模式
  python run_evoagentx_medical.py --demo                                    # 演示模式
  python run_evoagentx_medical.py --query "患者男性，58岁，体重下降，黄疸"    # 单次查询
  python run_evoagentx_medical.py --batch cases.txt                         # 批处理模式
  python run_evoagentx_medical.py --reindex                                 # 重建索引并进入交互模式
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["interactive", "demo", "batch"],
        default="interactive",
        help="运行模式 (默认: interactive)"
    )
    
    parser.add_argument(
        "--batch",
        type=str,
        help="批处理模式：指定包含症状描述的文本文件路径"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示模式"
    )
    
    parser.add_argument(
        "--reindex",
        action="store_true", 
        help="强制重建医学文档索引"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="直接输入症状描述进行单次分析"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    try:
        # 显示启动信息
        print("🏥 EvoAgentX医疗智能分析系统")
        print("=" * 50)
        print("⚠️  重要提醒：本系统仅供医疗专业人员参考，不能替代正式医疗诊断！")
        print("=" * 50)
        
        # 初始化执行器
        print("🚀 正在初始化EvoAgentX医疗工作流执行器...")
        executor = MedicalWorkflowExecutor()
        
        # 检查并建立索引
        print("📚 检查医学文档索引...")
        if not executor.ensure_rag_indexed(force_reindex=args.reindex):
            print("❌ 无法建立医学文档索引，程序退出")
            return 1
        
        print("✅ 系统初始化完成")
        
        # 根据参数确定运行模式
        if args.query:
            # 单次查询模式
            single_query_mode(executor, args.query)
        elif args.demo or args.mode == "demo":
            demo_mode(executor)
        elif args.batch or args.mode == "batch":
            batch_file = args.batch if args.batch else input("请输入病例文件路径: ")
            batch_mode(executor, batch_file)
        else:
            interactive_mode(executor)
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n👋 用户中断，程序退出")
        return 0
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
        print(f"❌ 程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())