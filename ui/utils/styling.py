from PyQt6.QtGui import QColor, QFont
from enum import Enum

class EconomicStateColors:
    """Colors for different economic states"""
    # Main colors - softer shades
    OVERHEATED = "#e53935"  # Less bright red
    GOLDILOCKS = "#43a047"  # Softer green
    DEPRESSION = "#1e88e5"  # Softer blue
    
    # Much lighter versions for backgrounds
    OVERHEATED_LIGHT = "#ffffff"  # White background instead of pink
    GOLDILOCKS_LIGHT = "#ffffff"  # White background instead of green
    DEPRESSION_LIGHT = "#ffffff"  # White background instead of blue
    
    # Metric colors - more readable
    METRIC_POSITIVE = "#2e7d32"  # Darker green for positive values
    METRIC_NEGATIVE = "#c62828"  # Darker red for negative values
    METRIC_NEUTRAL = "#424242"  # Dark gray for neutral values
    
    @classmethod
    def get_state_color(cls, state_value: float) -> str:
        """Get color based on economic state value"""
        if state_value > 0.3:
            return cls.OVERHEATED
        elif state_value < -0.3:
            return cls.DEPRESSION
        return cls.GOLDILOCKS

class ChartColors:
    """Colors for charts and graphs"""
    # Line colors
    HISTORICAL_LINE = "#666666"
    PREDICTION_LINE = "#999999"
    THRESHOLD_LINE = "#cccccc"
    
    # Background colors
    GRAPH_BACKGROUND = "#ffffff"
    PANEL_BACKGROUND = "#f5f5f5"
    
    # Grid colors
    MAJOR_GRID = "#e0e0e0"
    MINOR_GRID = "#f0f0f0"

class Fonts:
    """Font definitions for different UI elements"""
    TITLE = QFont("Arial", 16, QFont.Weight.Bold)
    SUBTITLE = QFont("Arial", 14, QFont.Weight.Bold)
    BODY = QFont("Arial", 12)
    SMALL = QFont("Arial", 10)
    
    # Monospace for numbers
    NUMBERS = QFont("Courier New", 12)

class StyleSheets:
    """Qt StyleSheet definitions"""
    
    # Main window
    WINDOW = """
        QMainWindow {
            background-color: #f5f5f5;
        }
    """
    
    # Panels
    PANEL = """
        QFrame {
            background-color: white;
            border: 1px solid #dcdcdc;
            border-radius: 4px;
        }
    """
    
    # State indicator panel
    STATE_PANEL = """
        QFrame#statePanel {
            padding: 10px;
            margin: 5px;
            background-color: white;
            border: 1px solid #dcdcdc;
            border-radius: 4px;
        }
    """
    
    # Prediction boxes
    PREDICTION_BOX = """
        QFrame#predictionBox {
            padding: 8px;
            margin: 4px;
            background-color: white;
            border: 1px solid #dcdcdc;
            border-radius: 4px;
        }
    """
    
    # Graph panel
    GRAPH_PANEL = """
        QFrame#graphPanel {
            background-color: white;
            border: 1px solid #dcdcdc;
            border-radius: 4px;
            padding: 10px;
        }
    """
    
    # Contributors panel
    CONTRIBUTORS_PANEL = """
        QFrame#contributorsPanel {
            padding: 10px;
            background-color: white;
            border: 1px solid #dcdcdc;
            border-radius: 4px;
        }
    """

class Dimensions:
    """UI element dimensions"""
    # Window
    MIN_WINDOW_WIDTH = 1200
    MIN_WINDOW_HEIGHT = 800
    
    # Panels
    STATE_PANEL_HEIGHT = 120
    PREDICTIONS_PANEL_HEIGHT = 150
    GRAPH_PANEL_HEIGHT = 400
    CONTRIBUTORS_PANEL_HEIGHT = 150
    
    # Margins and padding
    PANEL_MARGIN = 10
    PANEL_PADDING = 15
    
    # Graph
    GRAPH_LINE_WIDTH = 2
    THRESHOLD_LINE_WIDTH = 1
    
    @staticmethod
    def get_panel_margins():
        """Returns consistent panel margins"""
        return [10, 10, 10, 10]  # left, top, right, bottom

class Labels:
    """Text labels used throughout the UI"""
    WINDOW_TITLE = "Economic State Monitor"
    
    # Panel titles
    CURRENT_STATE = "Current Economic State"
    PREDICTIONS = "Forward Predictions"
    HISTORICAL = "Historical Trend"
    CONTRIBUTORS = "Top Contributors"
    
    # Time periods
    THREE_MONTH = "3 Months"
    SIX_MONTH = "6 Months"
    TWELVE_MONTH = "12 Months"
    
    # Axis labels
    TIME_AXIS = "Time"
    STATE_AXIS = "Economic State"
    
    # States
    OVERHEATED_LABEL = "OVERHEATED"
    GOLDILOCKS_LABEL = "GOLDILOCKS"
    DEPRESSION_LABEL = "DEPRESSION"

def get_severity_color(value: float, baseline: float = 0.5) -> str:
    """Returns a color based on the severity of a value"""
    intensity = min(abs(value / baseline), 1.0)
    
    if value > 0:
        # Red scale for positive values (overheated)
        return f"rgb({255},{int(255*(1-intensity))},{int(255*(1-intensity))})"
    else:
        # Blue scale for negative values (depression)
        return f"rgb({int(255*(1-intensity))},{int(255*(1-intensity))},{255})"