from typing import TypedDict,List,Set,Annotated
import operator

# 1. 定义全局状态
class RobotState(TypedDict):
    # 1.1 输入数据
    user_command:str    # 用户初始指令
    visual_input:str    # 图片路径或者描述

    # 1.2 语义上下文
    semantic_context:str# 从csv或者图片中提取的文字描述

    # 1.3 状态记忆
    short_term_memory:List[str] # 滑动窗口
    visualed_set:[List[str],operator.add]   # 长期记忆
    current_position:str        # 当前位置

    # 1.4 LLM输出
    next_target:str
    thought_process:str # CoT

    # 1.5 循环次数
    step_count:int

# 2. 短期记忆功能
def module_memory(state:RobotState):
    print("---更新记忆档案---")

    target = state["next_target"]
    current = state.get("short_term_memory",[])

    # 2.1 加入新目标
    new_short_term = current + [target]

    # 2.2 滑动窗口
    if len(new_short_term) > 3:
        new_short_term = new_short_term[-3:]

    # 2.3 返回记忆
    return {
        "short_term_memory":new_short_term,
        "visited_set":[target],
        "step_count":state.get("step_count",0) + 1
    }
