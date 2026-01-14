from langgraph.graph import StateGraph,END

from model3.Planner_model import module_planner
from model_1.Observer_model import module_perception
from model_2.State_model import RobotState, module_memory

# 1. 定义图纸
workflow = StateGraph(RobotState)

# 2. 加入节点
workflow.add_node("perception",module_perception)
workflow.add_node("plannar",module_planner)
workflow.add_node("memory",module_memory)

# 3. 流转关系
workflow.set_entry_point("perception")

workflow.add_edge("percetion","planner")
workflow.add_edge("planner","memory")

# 4. 闭环
def check_continue(state):
    steps = state.get("step_count",0)
    visited = state.get("visited_set",[])

    # 4.1 停止条件
    """ 如果走了5步,或者已经访问'exit'，就停止 """
    if steps >= 5 or "Exit" in visited:
        print("任务结束")
        return "end"
    else:
        print("继续下一帧预测")
        return "continue"
# 4.2 添加条件
workflow.add_conditional_edges(
    "memory",   # 前一个节点
    check_continue, # 判断
    {
        "continue":"perception" # 回到起点
        "end":END               # 结束
    }
)

def main():
    app = workflow.compile


if __name__ == "__main__":
    initial_state = {
    "short_term_memory": [],
    "visited_set": [],
    "step_count": 0
}
    main()
