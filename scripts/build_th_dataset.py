import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Exhibit:
    exhibit_id: str
    name: str
    exhibit_type: str
    visual_features: str
    predecessors: List[str]
    successors: List[str]


@dataclass
class GazeEvent:
    exhibit_id: str
    exhibit_name: str
    duration_ms: float
    fixation_count: int
    start_video_time: float
    end_video_time: float


class Topology:
    def __init__(self, map_file: str):
        self.exhibits: Dict[str, Exhibit] = {}
        self.load_topology(map_file)
    
    def load_topology(self, map_file: str):
        df = pd.read_excel(map_file)
        
        for _, row in df.iterrows():
            exhibit_id = str(row['ID']).strip()
            name = str(row['Name']).strip()
            exhibit_type = str(row['Type']).strip()
            visual_features = str(row['Visual_Features']).strip()
            neighbors = str(row['Neighbors (含关系)']).strip()
            
            predecessors = []
            successors = []
            
            if neighbors and neighbors != 'nan':
                neighbors = neighbors.replace('nan', '').strip()
                if neighbors:
                    successor_id = neighbors.strip()
                    successors.append(successor_id)
            
            self.exhibits[exhibit_id] = Exhibit(
                exhibit_id=exhibit_id,
                name=name,
                exhibit_type=exhibit_type,
                visual_features=visual_features,
                predecessors=predecessors,
                successors=successors
            )
    
    def get_exhibit(self, exhibit_id: str) -> Optional[Exhibit]:
        return self.exhibits.get(exhibit_id)
    
    def get_neighbors(self, exhibit_id: str, depth: int = 2) -> Dict[str, List[Exhibit]]:
        result = {
            'left': [],
            'right': []
        }
        
        exhibit = self.get_exhibit(exhibit_id)
        if not exhibit:
            return result
        
        visited = set()
        
        def bfs_successors(current_id: str, current_depth: int):
            if current_depth == 0 or current_id in visited:
                return
            visited.add(current_id)
            
            current_exhibit = self.get_exhibit(current_id)
            if current_exhibit:
                for succ_id in current_exhibit.successors:
                    succ_exhibit = self.get_exhibit(succ_id)
                    if succ_exhibit and succ_id not in visited:
                        result['right'].append(succ_exhibit)
                        bfs_successors(succ_id, current_depth - 1)
        
        def find_predecessors(target_id: str, current_depth: int):
            if current_depth == 0:
                return
            
            for other_id, other_exhibit in self.exhibits.items():
                if other_id in visited:
                    continue
                
                if target_id in other_exhibit.successors:
                    visited.add(other_id)
                    result['left'].append(other_exhibit)
                    find_predecessors(other_id, current_depth - 1)
        
        find_predecessors(exhibit_id, depth)
        visited.clear()
        bfs_successors(exhibit_id, depth)
        
        return result


class THDatasetBuilder:
    def __init__(self, map_file: str, data_dir: str, output_dir: str, use_cot: bool = False):
        self.topology = Topology(map_file)
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.use_cot = use_cot
        
        os.makedirs(output_dir, exist_ok=True)
    
    def process_all_data(self, history_windows: List[int] = [2, 3, 4]):
        text_samples = []
        vision_samples = []
        
        all_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if (file.endswith('.xlsx') or file.endswith('.csv')) and not file.startswith('~$'):
                    data_file = os.path.join(root, file)
                    all_files.append(data_file)
        
        print(f"找到的文件总数: {len(all_files)}")
        for i, f in enumerate(all_files, 1):
            print(f"  {i}. {os.path.basename(f)}")
        
        processed_count = 0
        for data_file in all_files:
            print(f"\n处理文件: {os.path.basename(data_file)}")
            
            events = self.process_user_data(data_file)
            
            if len(events) >= 3:
                text_batch = self.create_training_samples(events, history_windows, mode='text')
                vision_batch = self.create_training_samples(events, history_windows, mode='vision')
                text_samples.extend(text_batch)
                vision_samples.extend(vision_batch)
                processed_count += 1
                print(f"  生成样本数: {len(text_batch)} (text), {len(vision_batch)} (vision)")
            else:
                print(f"  事件数不足，跳过 (事件数: {len(events)})")
        
        print(f"\n成功处理文件数: {processed_count}/{len(all_files)}")
        print(f"Total text samples: {len(text_samples)}")
        print(f"Total vision samples: {len(vision_samples)}")
        
        output_text_file = os.path.join(self.output_dir, 'th_gaze_prediction_text.json')
        output_vision_file = os.path.join(self.output_dir, 'th_gaze_prediction_vision.json')
        
        with open(output_text_file, 'w', encoding='utf-8') as f:
            json.dump(text_samples, f, ensure_ascii=False, indent=2)
        
        with open(output_vision_file, 'w', encoding='utf-8') as f:
            json.dump(vision_samples, f, ensure_ascii=False, indent=2)
        
        print(f"Text dataset saved to: {output_text_file}")
        print(f"Vision dataset saved to: {output_vision_file}")
    
    def process_user_data(self, data_file: str) -> List[GazeEvent]:
        if data_file.lower().endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_excel(data_file)
        
        exhibit_records = df[df['Hot key'].notna()].copy()
        
        if len(exhibit_records) == 0:
            return []
        
        exhibit_records = exhibit_records.sort_values('Video Time[HH:mm:ss.ms]')
        
        def video_time_to_seconds(time_str):
            try:
                parts = time_str.split(':')
                if len(parts) == 3:
                    h, m, s = parts
                    s_parts = s.split('.')
                    seconds = int(h) * 3600 + int(m) * 60 + float(s)
                    return seconds
            except:
                return 0
            return 0
        
        exhibit_records['video_time_sec'] = exhibit_records['Video Time[HH:mm:ss.ms]'].apply(video_time_to_seconds)
        
        events = []
        i = 0
        while i < len(exhibit_records) - 1:
            record1 = exhibit_records.iloc[i]
            record2 = exhibit_records.iloc[i + 1]
            
            current_hotkey = str(record1['Hot key']).strip()
            next_hotkey = str(record2['Hot key']).strip()
            
            duration_ms = (record2['video_time_sec'] - record1['video_time_sec']) * 1000
            
            if current_hotkey == next_hotkey:
                exhibit = self.topology.get_exhibit(current_hotkey)
                if exhibit:
                    events.append(GazeEvent(
                        exhibit_id=current_hotkey,
                        exhibit_name=exhibit.name,
                        duration_ms=duration_ms,
                        fixation_count=1,
                        start_video_time=record1['video_time_sec'],
                        end_video_time=record2['video_time_sec']
                    ))
            
            else:
                exhibit1 = self.topology.get_exhibit(current_hotkey)
                if exhibit1:
                    events.append(GazeEvent(
                        exhibit_id=current_hotkey,
                        exhibit_name=exhibit1.name,
                        duration_ms=duration_ms,
                        fixation_count=1,
                        start_video_time=record1['video_time_sec'],
                        end_video_time=record2['video_time_sec']
                    ))
            
            i += 1
        
        return events
    
    def create_training_samples(
        self,
        events: List[GazeEvent],
        history_windows: List[int] = [2, 3, 4],
        mode: str = 'text'
    ) -> List[Dict]:
        samples = []
        
        for history_window in history_windows:
            min_length = history_window + 1
            if len(events) < min_length:
                continue
            
            stride = history_window
            
            for i in range(0, len(events) - min_length + 1, stride):
                history = events[i:i + history_window]
                future = events[i + history_window]
                
                last_event = history[-1]
                context = self.topology.get_neighbors(
                    last_event.exhibit_id,
                    depth=2
                )
                
                if mode == 'text':
                    sample = self.build_text_sample(history, future, context)
                else:
                    sample = self.build_vision_sample(history, future, context)
                
                samples.append(sample)
        
        return samples
    
    def build_text_sample(
        self,
        history_events: List[GazeEvent],
        future_event: GazeEvent,
        context_exhibits: Dict[str, List[Exhibit]]
    ) -> Dict:
        history_text = []
        for event in history_events:
            exhibit = self.topology.get_exhibit(event.exhibit_id)
            if exhibit:
                history_text.append(
                    f"{event.exhibit_name} - 停留{event.duration_ms/1000:.1f}秒"
                )
        
        context_text = []
        if context_exhibits.get('left'):
            for exhibit in context_exhibits['left'][:2]:
                context_text.append(f"前序展品: {exhibit.name}")
        if context_exhibits.get('right'):
            for exhibit in context_exhibits['right'][:2]:
                context_text.append(f"后序展品: {exhibit.name}")
        
        instruction = "根据历史观看序列和周边展品信息，预测用户接下来最有可能观看的下一个展品及其停留时间。"
        
        input_text = (
            f"历史观看序列:\n" + "\n".join(history_text) + "\n\n" +
            f"周边展品信息:\n" + "\n".join(context_text)
        )
        
        future_exhibit = self.topology.get_exhibit(future_event.exhibit_id)
        if future_exhibit:
            output_text = f"{future_exhibit.name} - 预计停留{future_event.duration_ms/1000:.1f}秒"
        else:
            output_text = f"{future_event.exhibit_id} - 预计停留{future_event.duration_ms/1000:.1f}秒"
        
        if self.use_cot:
            output_text = self._build_cot_output_text(
                history_events, future_event, context_exhibits, output_text
            )
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
    
    def build_vision_sample(
        self,
        history_events: List[GazeEvent],
        future_event: GazeEvent,
        context_exhibits: Dict[str, List[Exhibit]]
    ) -> Dict:
        scene_exhibits = []
        
        current_event = history_events[-1]
        current_exhibit = self.topology.get_exhibit(current_event.exhibit_id)
        if current_exhibit:
            scene_exhibits.append(f"当前位置: {self._simulate_vision_extraction(current_exhibit)}")
        
        if context_exhibits.get('left'):
            for exhibit in context_exhibits['left'][:2]:
                scene_exhibits.append(f"左侧: {self._simulate_vision_extraction(exhibit)}")
        
        if context_exhibits.get('right'):
            for exhibit in context_exhibits['right'][:2]:
                scene_exhibits.append(f"右侧: {self._simulate_vision_extraction(exhibit)}")
        
        instruction = (
            "从场景图片中识别出以下展品信息，"
            "预测用户接下来最有可能观看的下一个展品及其停留时间。"
        )
        
        input_text = (
            f"当前场景中的藏品:\n" + "\n".join(scene_exhibits)
        )
        
        future_exhibit = self.topology.get_exhibit(future_event.exhibit_id)
        if future_exhibit:
            output_text = f"{future_exhibit.name} - 预计停留{future_event.duration_ms/1000:.1f}秒"
        else:
            output_text = f"{future_event.exhibit_id} - 预计停留{future_event.duration_ms/1000:.1f}秒"
        
        if self.use_cot:
            output_text = self._build_cot_output_vision(
                history_events, future_event, scene_exhibits, output_text
            )
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
    
    def _simulate_vision_extraction(self, exhibit: Exhibit) -> str:
        if exhibit.exhibit_type == "Exhibit":
            return f"[展品]{exhibit.name}: {exhibit.visual_features.strip()}"
        elif exhibit.exhibit_type == "Text":
            return f"[文字]{exhibit.name}: {exhibit.visual_features.strip()}"
        else:
            return f"[区域]{exhibit.name}: {exhibit.visual_features.strip()}"
    
    def _build_cot_output_text(
        self,
        history_events: List[GazeEvent],
        future_event: GazeEvent,
        context_exhibits: Dict[str, List[Exhibit]],
        final_answer: str
    ) -> str:
        reasoning = []
        
        last_event = history_events[-1]
        reasoning.append(f"用户最后观看的是{last_event.exhibit_name}，停留了{last_event.duration_ms/1000:.1f}秒。")
        
        total_duration = sum(e.duration_ms for e in history_events) / 1000
        reasoning.append(f"历史观看总时长为{total_duration:.1f}秒，平均每个展品停留{total_duration/len(history_events):.1f}秒。")
        
        if context_exhibits.get('left'):
            left_names = [e.name for e in context_exhibits['left'][:2]]
            reasoning.append(f"根据拓扑关系，该展品的前序展品包括：{', '.join(left_names)}。")
        
        if context_exhibits.get('right'):
            right_names = [e.name for e in context_exhibits['right'][:2]]
            reasoning.append(f"根据拓扑关系，该展品的后序展品包括：{', '.join(right_names)}。")
        
        future_exhibit = self.topology.get_exhibit(future_event.exhibit_id)
        reasoning.append(f"综合考虑历史观看模式和拓扑关系，预测用户接下来最有可能观看{future_exhibit.name}。")
        
        reasoning.append(f"根据历史停留时间分布，预计停留{future_event.duration_ms/1000:.1f}秒。")
        
        reasoning.append(f"最终答案：{final_answer}")
        
        return "\n".join(reasoning)
    
    def _build_cot_output_vision(
        self,
        history_events: List[GazeEvent],
        future_event: GazeEvent,
        scene_exhibits: List[str],
        final_answer: str
    ) -> str:
        reasoning = []
        
        reasoning.append(f"从场景图片中识别到{len(scene_exhibits)}个展品/区域。")
        
        current_event = history_events[-1]
        reasoning.append(f"用户当前位置在{current_event.exhibit_name}。")
        
        reasoning.append(f"根据场景布局和展品分布，预测用户接下来最有可能观看{self.topology.get_exhibit(future_event.exhibit_id).name}。")
        
        reasoning.append(f"预计停留{future_event.duration_ms/1000:.1f}秒。")
        
        reasoning.append(f"最终答案：{final_answer}")
        
        return "\n".join(reasoning)


if __name__ == "__main__":
    map_file = "skills/topology/assets/TH.xlsx"
    data_dir = "data/TH"
    output_dir = "data/processed"
    
    builder = THDatasetBuilder(
        map_file=map_file,
        data_dir=data_dir,
        output_dir=output_dir,
        use_cot=False
    )
    
    builder.process_all_data(history_windows=[2, 3, 4])
