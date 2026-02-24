"""
Unit Tests for Semantic Segmentation Module

This module contains tests for the semantic segmentation components.
Due to the nature of Replicate API (requires actual API calls), most tests
use mocking to avoid unnecessary API costs.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.segmentation.segmenter import SemanticSegmenter
from config import ModelConfig


class TestSemanticSegmenter:
    """Test SemanticSegmenter class"""

    @pytest.fixture
    def model_config(self):
        """Create test model config"""
        return ModelConfig(
            replicate_api_token="test-token-r8_xxxxxxxxxxxxx",
            sam2_model_version="meta/sam2-hiera-large"
        )

    @pytest.fixture
    def segmenter(self, model_config):
        """Create SemanticSegmenter instance"""
        with patch('skills.segmentation.segmenter.replicate.Client'):
            segmenter = SemanticSegmenter(model_config)
            segmenter.client = Mock()
            return segmenter

    def test_initialization_with_token(self, model_config):
        """Test initialization with API token"""
        with patch('skills.segmentation.segmenter.replicate.Client') as mock_client:
            segmenter = SemanticSegmenter(model_config)
            assert segmenter.client is not None

    def test_initialization_without_token(self):
        """Test initialization without API token"""
        config = ModelConfig(replicate_api_token="")
        segmenter = SemanticSegmenter(config)
        assert segmenter.client is None

    def test_segment_image_without_client(self):
        """Test segmentation without client (no API token)"""
        # Create config without token
        config = ModelConfig(replicate_api_token="")
        segmenter = SemanticSegmenter(config)

        result = segmenter.segment_image("test.jpg")

        assert "error" in result
        assert "not initialized" in result['error'].lower()

    @patch('os.path.exists')
    def test_segment_image_file_not_found(self, mock_exists, segmenter):
        """Test segmentation with non-existent file"""
        mock_exists.return_value = False
        segmenter.client = Mock()  # Mock the client

        result = segmenter.segment_image("nonexistent.jpg")

        assert "error" in result
        assert "not found" in result['error'].lower()

    def test_parse_replicate_output_dict(self, segmenter):
        """Test parsing Replicate output (dict format)"""
        output = {
            'masks': [[1, 0], [0, 1]],
            'scores': [0.95, 0.87]
        }

        result = segmenter._parse_replicate_output(output)

        assert 'masks' in result
        assert 'scores' in result
        assert 'boxes' in result
        assert 'centers' in result

    def test_parse_replicate_output_list(self, segmenter):
        """Test parsing Replicate output (list format)"""
        output = [[1, 0], [0, 1]]

        result = segmenter._parse_replicate_output(output)

        assert 'masks' in result
        assert len(result['masks']) == 2

    def test_extract_boxes_and_centers(self, segmenter):
        """Test box and center extraction"""
        # Create a simple 2x2 mask
        masks = [
            np.array([[1, 1], [1, 1]]),
            np.array([[0, 1], [0, 1]])
        ]

        boxes, centers = segmenter._extract_boxes_and_centers(masks)

        assert len(boxes) == 2
        assert len(centers) == 2
        # First mask covers full 2x2 area
        assert boxes[0] == (0, 0, 1, 1)
        assert centers[0] == (0, 0)  # Center of 2x2

    def test_prepare_for_vlm(self, segmenter):
        """Test VLM preparation"""
        segmentation_result = {
            'masks': [[1, 0]],
            'scores': [0.95],
            'boxes': [(0, 0, 100, 100)],
            'centers': [(50, 50)]
        }

        # Create mock cropped images
        mock_image = Image.new('RGB', (100, 100), color='red')
        segmentation_result['cropped_images'] = [mock_image]

        vlm_inputs = segmenter.prepare_for_vlm(segmentation_result)

        assert len(vlm_inputs) == 1
        assert vlm_inputs[0]['center'] == (50, 50)
        assert vlm_inputs[0]['confidence'] == 0.95
        assert 'image' in vlm_inputs[0]

    @patch('builtins.open')
    @patch('os.makedirs')
    def test_export_masks(self, mock_makedirs, mock_open, segmenter):
        """Test mask export"""
        segmentation_result = {
            'masks': [
                np.array([[1, 0], [0, 1]]),
                np.array([[0, 1], [1, 0]])
            ]
        }

        # Mock PIL Image
        with patch('skills.segmentation.segmenter.Image.fromarray') as mock_fromarray:
            mock_image = Mock()
            mock_fromarray.return_value = mock_image

            paths = segmenter.export_masks(segmentation_result, "output_dir")

            assert len(paths) == 2
            mock_image.save.assert_called()


class TestMaskProcessing:
    """Test mask processing utilities"""

    @pytest.fixture
    def model_config(self):
        return ModelConfig(replicate_api_token="test-token")

    @pytest.fixture
    def segmenter(self, model_config):
        with patch('skills.segmentation.segmenter.replicate.Client'):
            return SemanticSegmenter(model_config)

    def test_empty_mask_list(self, segmenter):
        """Test handling empty mask list"""
        boxes, centers = segmenter._extract_boxes_and_centers([])

        assert boxes == []
        assert centers == []

    def test_single_pixel_mask(self, segmenter):
        """Test mask with single pixel"""
        masks = [np.array([[1]])]

        boxes, centers = segmenter._extract_boxes_and_centers(masks)

        assert len(boxes) == 1
        assert len(centers) == 1
        assert boxes[0] == (0, 0, 0, 0)
        assert centers[0] == (0, 0)

    def test_irregular_mask(self, segmenter):
        """Test irregular shaped mask"""
        # Create an L-shaped mask
        masks = [np.array([
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1]
        ])]

        boxes, centers = segmenter._extract_boxes_and_centers(masks)

        # Should cover the full extent of the mask
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert x1 >= 0 and y1 >= 0
        assert x2 > x1 and y2 > y1


class TestSegmentationIntegration:
    """Integration tests for segmentation workflow"""

    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            replicate_api_token="test-token",
            sam2_model_version="meta/sam2-hiera-large"
        )

    def test_full_segmentation_workflow(self, model_config):
        """Test complete segmentation workflow with mocks"""
        with patch('skills.segmentation.segmenter.replicate.Client') as mock_client_class:
            # Setup mocks
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock Replicate API response
            mock_client.run.return_value = {
                'masks': [
                    np.array([[1, 1], [1, 1]]),
                    np.array([[0, 1], [1, 0]])
                ],
                'scores': [0.95, 0.87]
            }

            with patch('os.path.exists', return_value=True):
                segmenter = SemanticSegmenter(model_config)

                # Mock image loading
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__ = Mock()
                    mock_open.return_value.__exit__ = Mock()
                    mock_open.return_value.read = Mock()

                    result = segmenter.segment_image("test.jpg")

                    # Should have called the API
                    assert mock_client.run.called

                    # Should return structured result
                    assert 'masks' in result or 'error' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
