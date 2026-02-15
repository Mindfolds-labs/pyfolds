"""Tests for type utilities."""

import pyfolds
from pyfolds.utils.types import LearningMode, ConnectionType


class TestTypes:
    """Test types and enums."""
    
    def test_learning_mode(self):
        """Test learning mode enum."""
        assert LearningMode.ONLINE.value == "online"
        assert LearningMode.ONLINE.is_learning() is True
        assert LearningMode.SLEEP.is_consolidating() is True
    
    def test_connection_type(self):
        """Test connection type enum."""
        assert ConnectionType.DENSE.value == "dense"
        assert ConnectionType.SPARSE.value == "sparse"
