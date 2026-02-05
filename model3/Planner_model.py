from langchain_core.messages import HumanMessage

from model_2.State_model import RobotState

def module_planner(state:RobotState):
    print("---gaze正在预测下一步---")

    # 1. 获取当前状态，构建prompt
    context = state.get("semantic_context","无")
    visited = state.get("visualed_set",[])
    recent = state.get("short_term_memory",[])

    prompt = f"""
    你是一位gaze导航助手。
    [环境]:{context}
    [记忆]:已访问{visited},最近看了{recent}
    [任务]:请预测下一个去的物体ID。
    """