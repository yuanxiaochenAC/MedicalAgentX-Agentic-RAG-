#!/usr/bin/env python3
"""调试Agent名称和调用问题的脚本"""

import sys
import traceback
from pathlib import Path

# 添加app路径
sys.path.append(str(Path(__file__).parent / "app"))

def debug_agents():
    """调试Agent相关问题"""
    try:
        print("🔍 开始调试Agent问题...")
        
        # 1. 导入工作流执行器
        from evoagentx_medical_workflow import MedicalWorkflowExecutor
        
        print("✅ 成功导入MedicalWorkflowExecutor")
        
        # 2. 初始化执行器
        print("🚀 正在初始化执行器...")
        executor = MedicalWorkflowExecutor()
        
        print("✅ 执行器初始化完成")
        
        # 3. 检查AgentManager和Agents
        if executor.agent_manager:
            print(f"📊 AgentManager已创建，包含 {len(executor.agent_manager.agents)} 个Agent")
            
            print("\n📝 可用的Agent列表:")
            agent_names = executor.agent_manager.list_agents()
            for i, name in enumerate(agent_names, 1):
                print(f"  {i}. {name}")
            
            # 4. 测试Agent获取
            print("\n🧪 测试Agent获取:")
            test_names = ["MedicalReasoner", "MedicalreasonerAgent", "reasoner", "MedicalReasoner"]
            
            for test_name in test_names:
                try:
                    agent = executor._get_agent_by_name(test_name)
                    if agent:
                        print(f"  ✅ 找到Agent: {test_name} -> {agent.name}")
                    else:
                        print(f"  ❌ 未找到Agent: {test_name}")
                except Exception as e:
                    print(f"  ❌ 获取Agent '{test_name}' 时出错: {str(e)}")
            
            # 5. 测试真正的Agent调用
            print("\n🎯 测试Agent调用:")
            if agent_names:
                first_agent_name = agent_names[0]
                try:
                    agent = executor.agent_manager.get_agent(first_agent_name)
                    print(f"  ✅ 成功获取Agent: {first_agent_name}")
                    print(f"  Agent类型: {type(agent)}")
                    print(f"  Agent描述: {getattr(agent, 'description', 'N/A')}")
                    
                    # 测试简单调用
                    test_inputs = {
                        "symptom_text": "测试症状",
                        "similar_cases": "测试相似病例",
                        "retrieval_metadata": "测试元数据"
                    }
                    
                    print(f"  🧪 尝试调用Agent...")
                    response = agent(inputs=test_inputs)
                    print(f"  ✅ Agent调用成功")
                    print(f"  响应类型: {type(response)}")
                    
                    if hasattr(response, 'content'):
                        print(f"  响应内容类型: {type(response.content)}")
                        if hasattr(response.content, 'content'):
                            content = response.content.content
                            print(f"  响应内容预览: {content[:100]}...")
                        else:
                            print(f"  响应内容: {response.content}")
                    else:
                        print(f"  响应: {response}")
                        
                except Exception as e:
                    print(f"  ❌ Agent调用失败: {str(e)}")
                    traceback.print_exc()
        else:
            print("❌ AgentManager未初始化")
            
    except Exception as e:
        print(f"❌ 调试过程中发生错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_agents()