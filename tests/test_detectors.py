import pytest
import numpy as np
from src.detectors import ColorCubeDetector
from src.utils import mock_frame, load_config

@pytest.fixture
def config():
    return load_config()

def test_color_detection(config):
    detector = ColorCubeDetector(config)
    frame = mock_frame()
    # Simulate blue patch
    frame[200:300, 200:300] = [255, 0, 0]  # BGR blue
    verts, contour = detector.detect(frame)
    assert contour is not None
    assert len(verts) == 4  # Should find quad

if __name__ == '__main__':
    pytest.main(['-v'])