"""
Unit Tests for Memory Management System

This module contains tests for the memory management components including
MemoryManager, ShortTermMemory, and LongTermMemory.
"""

import pytest
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.memory.manager import MemoryManager, GazeRecord
from skills.memory.short_term import ShortTermMemory
from skills.memory.long_term import LongTermMemory


class TestGazeRecord:
    """Test GazeRecord dataclass"""

    def test_create_gaze_record(self):
        """Test creating a gaze record"""
        record = GazeRecord(
            exhibit_id="TH-E01",
            exhibit_name="欢迎致辞",
            timestamp=datetime.now(),
            attention_level="A",
            estimated_duration=120
        )

        assert record.exhibit_id == "TH-E01"
        assert record.exhibit_name == "欢迎致辞"
        assert record.attention_level == "A"
        assert record.estimated_duration == 120
        assert record.actual_duration is None

    def test_to_dict(self):
        """Test converting GazeRecord to dictionary"""
        record = GazeRecord(
            exhibit_id="TH-B02",
            exhibit_name="何尊",
            timestamp=datetime.now(),
            attention_level="B",
            estimated_duration=60
        )

        data = record.to_dict()

        assert data['exhibit_id'] == "TH-B02"
        assert data['exhibit_name'] == "何尊"
        assert 'timestamp' in data
        assert data['attention_level'] == "B"


class TestShortTermMemory:
    """Test ShortTermMemory class"""

    def test_initialization(self):
        """Test ShortTermMemory initialization"""
        memory = ShortTermMemory(max_size=5)
        assert memory.max_size == 5
        assert len(memory) == 0
        assert memory.is_empty()

    def test_add_single_record(self):
        """Test adding a single record"""
        memory = ShortTermMemory(max_size=5)
        record = GazeRecord(
            exhibit_id="TH-E01",
            exhibit_name="Test",
            timestamp=datetime.now(),
            attention_level="A",
            estimated_duration=120
        )

        memory.add(record)
        assert len(memory) == 1
        assert not memory.is_empty()

    def test_sliding_window(self):
        """Test that sliding window works correctly"""
        memory = ShortTermMemory(max_size=3)

        # Add 5 records
        for i in range(5):
            record = GazeRecord(
                exhibit_id=f"TH-{i}",
                exhibit_name=f"Test {i}",
                timestamp=datetime.now(),
                attention_level="A",
                estimated_duration=120
            )
            memory.add(record)

        # Should only keep last 3
        assert len(memory) == 3
        recent = memory.get_recent()
        assert recent[0].exhibit_id == "TH-2"
        assert recent[1].exhibit_id == "TH-3"
        assert recent[2].exhibit_id == "TH-4"

    def test_get_recent(self):
        """Test getting recent records"""
        memory = ShortTermMemory(max_size=10)

        for i in range(5):
            record = GazeRecord(
                exhibit_id=f"TH-{i}",
                exhibit_name=f"Test {i}",
                timestamp=datetime.now(),
                attention_level="A",
                estimated_duration=120
            )
            memory.add(record)

        recent = memory.get_recent(3)
        assert len(recent) == 3
        assert recent[0].exhibit_id == "TH-2"

    def test_clear(self):
        """Test clearing memory"""
        memory = ShortTermMemory(max_size=5)
        record = GazeRecord(
            exhibit_id="TH-E01",
            exhibit_name="Test",
            timestamp=datetime.now(),
            attention_level="A",
            estimated_duration=120
        )

        memory.add(record)
        assert len(memory) == 1

        memory.clear()
        assert len(memory) == 0
        assert memory.is_empty()


class TestLongTermMemory:
    """Test LongTermMemory class"""

    def test_initialization(self):
        """Test LongTermMemory initialization"""
        memory = LongTermMemory(max_size=100)
        assert memory.max_size == 100
        assert len(memory) == 0

    def test_add_and_count(self):
        """Test adding records and counting"""
        memory = LongTermMemory()

        # Add records for same exhibit
        for i in range(3):
            record = GazeRecord(
                exhibit_id="TH-E01",
                exhibit_name="Test",
                timestamp=datetime.now(),
                attention_level="A",
                estimated_duration=120
            )
            memory.add(record)

        assert len(memory) == 3
        assert memory.get_visit_count("TH-E01") == 3
        assert memory.unique_count() == 1

    def test_visit_count_tracking(self):
        """Test visit count tracking"""
        memory = LongTermMemory()

        # Add different exhibits
        exhibits = ["TH-E01", "TH-B02", "TH-E01", "TH-B02", "TH-B02"]
        for exhibit_id in exhibits:
            record = GazeRecord(
                exhibit_id=exhibit_id,
                exhibit_name="Test",
                timestamp=datetime.now(),
                attention_level="A",
                estimated_duration=120
            )
            memory.add(record)

        assert memory.get_visit_count("TH-E01") == 2
        assert memory.get_visit_count("TH-B02") == 3
        assert memory.unique_count() == 2

    def test_most_visited(self):
        """Test most visited function"""
        memory = LongTermMemory()

        # Add records with different frequencies
        for i in range(5):
            record = GazeRecord(
                exhibit_id="TH-E01",
                exhibit_name="Exhibit 1",
                timestamp=datetime.now(),
                attention_level="A",
                estimated_duration=120
            )
            memory.add(record)

        for i in range(3):
            record = GazeRecord(
                exhibit_id="TH-B02",
                exhibit_name="Exhibit 2",
                timestamp=datetime.now(),
                attention_level="A",
                estimated_duration=120
            )
            memory.add(record)

        most_visited = memory.most_visited(2)
        assert len(most_visited) == 2
        assert most_visited[0] == ("TH-E01", 5)
        assert most_visited[1] == ("TH-B02", 3)

    def test_total_duration(self):
        """Test total duration calculation"""
        memory = LongTermMemory()

        # Add records with different durations
        durations = [120, 60, 30]
        for duration in durations:
            record = GazeRecord(
                exhibit_id="TH-E01",
                exhibit_name="Test",
                timestamp=datetime.now(),
                attention_level="A",
                estimated_duration=duration
            )
            memory.add(record)

        assert memory.total_duration() == sum(durations)

    def test_has_visited(self):
        """Test has_visited method"""
        memory = LongTermMemory()

        assert not memory.has_visited("TH-E01")

        record = GazeRecord(
            exhibit_id="TH-E01",
            exhibit_name="Test",
            timestamp=datetime.now(),
            attention_level="A",
            estimated_duration=120
        )
        memory.add(record)

        assert memory.has_visited("TH-E01")
        assert not memory.has_visited("TH-B02")


class TestMemoryManager:
    """Test MemoryManager class"""

    def test_initialization(self):
        """Test MemoryManager initialization"""
        manager = MemoryManager(short_term_size=5, long_term_max_size=100)
        assert len(manager.short_term) == 0
        assert len(manager.long_term) == 0

    def test_add_gaze(self):
        """Test adding gaze records"""
        manager = MemoryManager()
        record = GazeRecord(
            exhibit_id="TH-E01",
            exhibit_name="Test",
            timestamp=datetime.now(),
            attention_level="A",
            estimated_duration=120
        )

        manager.add_gaze(record)

        assert len(manager.short_term) == 1
        assert len(manager.long_term) == 1

    def test_add_observation(self):
        """Test adding observation directly"""
        manager = MemoryManager()
        manager.add_observation(
            exhibit_id="TH-E01",
            exhibit_name="Test",
            attention_level="B"
        )

        recent = manager.get_recent(1)
        assert len(recent) == 1
        assert recent[0].exhibit_id == "TH-E01"
        assert recent[0].estimated_duration == 60  # B level duration

    def test_get_statistics(self):
        """Test getting statistics"""
        manager = MemoryManager()

        # Add multiple observations
        manager.add_observation("TH-E01", "Exhibit 1", "A")
        manager.add_observation("TH-B02", "Exhibit 2", "B")
        manager.add_observation("TH-E01", "Exhibit 1", "C")

        stats = manager.get_statistics()

        assert stats['short_term_count'] == 3
        assert stats['long_term_count'] == 3
        assert stats['unique_exhibits'] == 2
        assert 'most_visited' in stats
        assert stats['total_duration'] > 0

    def test_has_visited(self):
        """Test has_visited through manager"""
        manager = MemoryManager()

        assert not manager.has_visited("TH-E01")

        manager.add_observation("TH-E01", "Test", "A")

        assert manager.has_visited("TH-E01")

    def test_clear(self):
        """Test clearing all memory"""
        manager = MemoryManager()
        manager.add_observation("TH-E01", "Test", "A")

        assert len(manager) == 1

        manager.clear()

        assert len(manager) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
