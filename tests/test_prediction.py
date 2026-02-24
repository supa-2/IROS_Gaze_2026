"""
Unit Tests for Prediction Engine

This module contains tests for the prediction engine components including
ContextBuilder, LLMReasoner, SequencePredictor, and PredictionEngine.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.memory.manager import GazeRecord
from skills.prediction.context_builder import ContextBuilder
from skills.prediction.llm_reasoner import LLMReasoner
from skills.prediction.sequence_predictor import SequencePredictor
from skills.prediction.engine import PredictionEngine
from config import Config, ModelConfig, AttentionConfig


class TestContextBuilder:
    """Test ContextBuilder class"""

    @pytest.fixture
    def topology_engine(self):
        """Create a mock topology engine"""
        engine = Mock()
        engine.graph = Mock()
        engine.graph.nodes = {
            'TH-E01': {'name': 'Welcome', 'features': 'Entrance area'},
            'TH-B02': {'name': 'He Zun', 'features': 'Bronze vessel'}
        }
        engine.map_name = 'TH'

        engine.query_node = Mock(return_value={
            'map': 'TH',
            'id': 'TH-E01',
            'info': {
                'name': 'Welcome',
                'type': 'Entrance',
                'features': 'Entrance area with introduction'
            },
            'context': {
                'previous_path': [],
                'next_path': [
                    {'id': 'TH-I-B01', 'name': 'Info Board', 'dist': 1}
                ],
                'direct_choices': [
                    {'id': 'TH-I-B01', 'relation': 'next'},
                    {'id': 'TH-B02', 'relation': 'visual'}
                ]
            }
        })

        return engine

    @pytest.fixture
    def context_builder(self, topology_engine):
        """Create ContextBuilder instance"""
        return ContextBuilder(topology_engine)

    @pytest.fixture
    def sample_gaze_record(self):
        """Create sample gaze record"""
        return GazeRecord(
            exhibit_id="TH-E01",
            exhibit_name="Welcome",
            timestamp=datetime.now(),
            attention_level="A",
            estimated_duration=120
        )

    def test_build_context(self, context_builder, sample_gaze_record):
        """Test building prediction context"""
        history = [sample_gaze_record]
        context = context_builder.build_context(sample_gaze_record, history)

        assert 'current' in context
        assert 'history' in context
        assert 'spatial' in context
        assert 'statistics' in context

        assert context['current']['id'] == "TH-E01"
        assert context['current']['name'] == "Welcome"
        assert len(context['history']) == 1

    def test_summarize_history(self, context_builder):
        """Test history summarization"""
        records = [
            GazeRecord("TH-E01", "Exhibit 1", datetime.now(), "A", 120),
            GazeRecord("TH-B02", "Exhibit 2", datetime.now(), "B", 60),
            GazeRecord("TH-C03", "Exhibit 3", datetime.now(), "C", 30)
        ]

        summary = context_builder._summarize_history(records)

        assert len(summary) == 3
        assert summary[0]['id'] == "TH-E01"
        assert summary[0]['level'] == "A"
        assert summary[0]['duration'] == 120

    def test_format_context_for_llm(self, context_builder, sample_gaze_record):
        """Test LLM-friendly formatting"""
        history = [sample_gaze_record]
        context = context_builder.build_context(sample_gaze_record, history)

        formatted = context_builder.format_context_for_llm(context)

        assert isinstance(formatted, str)
        assert "TH-E01" in formatted
        assert "Welcome" in formatted
        assert "A" in formatted


class TestLLMReasoner:
    """Test LLMReasoner class"""

    @pytest.fixture
    def model_config(self):
        """Create test model config"""
        return ModelConfig(
            llm_model="gpt-4o",
            llm_api_key="test-key",
            llm_temperature=0.7
        )

    @pytest.fixture
    def llm_reasoner(self, model_config):
        """Create LLMReasoner instance"""
        return LLMReasoner(model_config)

    @pytest.fixture
    def sample_context(self):
        """Create sample context"""
        return {
            'current': {
                'id': 'TH-E01',
                'name': 'Welcome',
                'features': 'Entrance area',
                'attention_level': 'A',
                'estimated_duration': 120
            },
            'history': [
                {'id': 'TH-E01', 'name': 'Welcome', 'level': 'A', 'duration': 120}
            ],
            'spatial': {
                'previous_path': [],
                'next_path': [],
                'reachable_options': [
                    {'id': 'TH-B02', 'name': 'He Zun', 'relation': 'next'}
                ]
            },
            'statistics': {
                'total_gazes': 1,
                'unique_exhibits': 1,
                'visited_exhibits': ['TH-E01']
            }
        }

    @pytest.fixture
    def attention_config(self):
        """Create attention config"""
        return AttentionConfig()

    def test_get_system_prompt(self, llm_reasoner, attention_config):
        """Test system prompt generation"""
        prompt = llm_reasoner._get_system_prompt(attention_config)

        assert isinstance(prompt, str)
        assert "Space Syntax Expert" in prompt
        assert "120" in prompt  # Duration for level A

    def test_format_user_prompt(self, llm_reasoner, sample_context):
        """Test user prompt formatting"""
        prompt = llm_reasoner._format_user_prompt(sample_context)

        assert isinstance(prompt, str)
        assert "TH-E01" in prompt
        assert "Welcome" in prompt

    def test_parse_llm_output_valid_json(self, llm_reasoner):
        """Test parsing valid JSON output"""
        json_output = '''{
  "prediction_id": "TH-B02",
  "prediction_name": "He Zun",
  "attention_level": "B",
  "estimated_duration": 60,
  "confidence": 0.85,
  "reasoning": "User is following the main path."
}'''

        result = llm_reasoner._parse_llm_output(json_output)

        assert result['prediction_id'] == "TH-B02"
        assert result['prediction_name'] == "He Zun"
        assert result['attention_level'] == "B"
        assert result['confidence'] == 0.85

    def test_parse_llm_output_with_code_block(self, llm_reasoner):
        """Test parsing JSON wrapped in code blocks"""
        output = '''Here's my prediction:

```json
{
  "prediction_id": "TH-B02",
  "prediction_name": "He Zun",
  "attention_level": "B",
  "estimated_duration": 60,
  "confidence": 0.85,
  "reasoning": "Following the path."
}
```

That's my prediction!'''

        result = llm_reasoner._parse_llm_output(output)

        assert result['prediction_id'] == "TH-B02"
        assert 'error' not in result

    def test_parse_llm_output_invalid(self, llm_reasoner):
        """Test parsing invalid output"""
        result = llm_reasoner._parse_llm_output("This is not JSON")

        assert 'error' in result
        assert result['prediction_id'] is None


class TestSequencePredictor:
    """Test SequencePredictor class"""

    @pytest.fixture
    def mock_topology(self):
        """Create mock topology engine"""
        engine = Mock()
        engine.query_node = Mock(side_effect=lambda x: {
            'id': x,
            'info': {'name': f'Exhibit {x}', 'features': 'Test'},
            'context': {
                'previous_path': [],
                'next_path': [],
                'direct_choices': []
            }
        })
        return engine

    @pytest.fixture
    def mock_reasoner(self):
        """Create mock reasoner that returns different predictions"""
        reasoner = Mock()
        # Create a side effect that returns different predictions each call
        predictions = [
            {
                'prediction_id': 'TH-B02',
                'prediction_name': 'He Zun',
                'attention_level': 'B',
                'estimated_duration': 60,
                'confidence': 0.8,
                'reasoning': 'Test reasoning 1'
            },
            {
                'prediction_id': 'TH-C03',
                'prediction_name': 'Exhibit 3',
                'attention_level': 'C',
                'estimated_duration': 30,
                'confidence': 0.7,
                'reasoning': 'Test reasoning 2'
            },
            {
                'prediction_id': 'TH-D04',
                'prediction_name': 'Exhibit 4',
                'attention_level': 'D',
                'estimated_duration': 15,
                'confidence': 0.6,
                'reasoning': 'Test reasoning 3'
            },
            {
                'prediction_id': 'TH-E05',
                'prediction_name': 'Exhibit 5',
                'attention_level': 'E',
                'estimated_duration': 5,
                'confidence': 0.5,
                'reasoning': 'Test reasoning 4'
            },
            {
                'prediction_id': 'TH-F06',
                'prediction_name': 'Exhibit 6',
                'attention_level': 'A',
                'estimated_duration': 120,
                'confidence': 0.9,
                'reasoning': 'Test reasoning 5'
            },
        ]
        reasoner.predict_next = Mock(side_effect=predictions)
        return reasoner

    @pytest.fixture
    def attention_config(self):
        """Create attention config"""
        return AttentionConfig()

    @pytest.fixture
    def sequence_predictor(self, mock_reasoner, mock_topology, attention_config):
        """Create SequencePredictor instance"""
        return SequencePredictor(mock_reasoner, mock_topology, attention_config)

    def test_predict_sequence(self, sequence_predictor):
        """Test multi-step sequence prediction"""
        context = {
            'current': {'id': 'TH-E01', 'name': 'Start', 'features': '', 'attention_level': 'A', 'estimated_duration': 120},
            'history': [],
            'spatial': {'previous_path': [], 'next_path': [], 'reachable_options': []},
            'statistics': {'total_gazes': 0, 'unique_exhibits': 0, 'visited_exhibits': []}
        }

        sequence = sequence_predictor.predict_sequence(context, n_steps=3)

        assert len(sequence) == 3
        assert sequence[0]['step_number'] == 1
        assert sequence[1]['step_number'] == 2
        assert sequence[2]['step_number'] == 3

    def test_format_sequence_summary(self, sequence_predictor):
        """Test sequence summary formatting"""
        sequence = [
            {
                'step_number': 1,
                'prediction_name': 'Exhibit 1',
                'prediction_id': 'TH-E01',
                'attention_level': 'A',
                'estimated_duration': 120,
                'confidence': 0.9
            },
            {
                'step_number': 2,
                'prediction_name': 'Exhibit 2',
                'prediction_id': 'TH-B02',
                'attention_level': 'B',
                'estimated_duration': 60,
                'confidence': 0.8
            }
        ]

        summary = sequence_predictor.format_sequence_summary(sequence)

        assert isinstance(summary, str)
        assert "Exhibit 1" in summary
        assert "TH-E01" in summary
        assert "180" in summary  # Total duration


class TestPredictionEngine:
    """Test PredictionEngine class"""

    @pytest.fixture
    def config(self):
        """Create test config"""
        config = Config()
        # Override with test values
        config.model.llm_api_key = "test-key"
        return config

    @pytest.fixture
    def prediction_engine(self, config):
        """Create PredictionEngine instance"""
        # Use patch to avoid actual topology loading
        with patch('skills.prediction.engine.TopologyEngine') as mock_topology:
            mock_topology.return_value.graph = Mock()
            mock_topology.return_value.graph.number_of_nodes = Mock(return_value=10)
            engine = PredictionEngine(config, map_name='TH')
            engine.topology = mock_topology.return_value
            return engine

    def test_add_observation(self, prediction_engine):
        """Test adding observation"""
        record = prediction_engine.add_observation(
            exhibit_id="TH-E01",
            exhibit_name="Welcome",
            attention_level="A"
        )

        assert record.exhibit_id == "TH-E01"
        assert record.estimated_duration == 120  # Level A duration
        assert len(prediction_engine.memory) == 1

    def test_get_memory_statistics(self, prediction_engine):
        """Test getting memory statistics"""
        prediction_engine.add_observation("TH-E01", "Exhibit 1", "A")
        prediction_engine.add_observation("TH-B02", "Exhibit 2", "B")

        stats = prediction_engine.get_memory_statistics()

        assert stats['short_term_count'] == 2
        assert stats['unique_exhibits'] == 2

    def test_has_visited(self, prediction_engine):
        """Test has_visited method"""
        assert not prediction_engine.has_visited("TH-E01")

        prediction_engine.add_observation("TH-E01", "Test", "A")

        assert prediction_engine.has_visited("TH-E01")

    def test_clear_memory(self, prediction_engine):
        """Test clearing memory"""
        prediction_engine.add_observation("TH-E01", "Test", "A")
        assert len(prediction_engine) == 1

        prediction_engine.clear_memory()
        assert len(prediction_engine) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
