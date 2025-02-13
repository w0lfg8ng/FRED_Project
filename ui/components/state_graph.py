from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QMargins  # Added QMargins here
from PyQt6.QtCharts import (QChart, QChartView, QLineSeries, QValueAxis,
                           QDateTimeAxis, QScatterSeries)
from PyQt6.QtGui import QPen, QColor, QFont, QPainter 
from PyQt6.QtGui import QPen, QColor, QFont
from datetime import datetime, timedelta
import time
from ..utils.styling import (
    EconomicStateColors,
    ChartColors,
    Fonts,
    StyleSheets,
    Dimensions,
    Labels
)

class StateGraph(QFrame):
    """Graph showing historical economic states and predictions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("graphPanel")
        self.setStyleSheet(StyleSheets.GRAPH_PANEL)
        self.setMinimumHeight(Dimensions.GRAPH_PANEL_HEIGHT)
        
        # Initialize chart components
        self._chart = QChart()
        self._chart_view = None
        self._historical_series = QLineSeries()
        self._prediction_series = QLineSeries()
        self._current_point_series = QScatterSeries()
        
        # Initialize axes
        self._time_axis = QDateTimeAxis()
        self._state_axis = QValueAxis()
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI layout and components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*Dimensions.get_panel_margins())
        
        # Title
        title_label = QLabel(Labels.HISTORICAL)
        title_label.setFont(Fonts.TITLE)
        layout.addWidget(title_label)
        
        # Set up chart
        self._setup_chart()
        self._setup_axes()
        self._setup_threshold_lines()
        
        # Create chart view
        self._chart_view = QChartView(self._chart)
        self._chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self._chart_view)
        
    def _setup_chart(self):
        """Configure the main chart settings."""
        self._chart.setBackgroundVisible(False)
        self._chart.setMargins(QMargins(20, 20, 20, 20))
        
        # Set up series
        self._historical_series.setName("Historical")
        pen = QPen(QColor(ChartColors.HISTORICAL_LINE))
        pen.setWidth(Dimensions.GRAPH_LINE_WIDTH)
        self._historical_series.setPen(pen)
        
        self._prediction_series.setName("Prediction")
        pred_pen = QPen(QColor(ChartColors.PREDICTION_LINE))
        pred_pen.setWidth(Dimensions.GRAPH_LINE_WIDTH)
        pred_pen.setStyle(Qt.PenStyle.DashLine)
        self._prediction_series.setPen(pred_pen)
        
        # Current point marker
        self._current_point_series.setMarkerSize(10)
        self._current_point_series.setColor(QColor(ChartColors.HISTORICAL_LINE))
        
        # Add series to chart
        self._chart.addSeries(self._historical_series)
        self._chart.addSeries(self._prediction_series)
        self._chart.addSeries(self._current_point_series)
        
    def _setup_axes(self):
        """Configure chart axes."""
        # Time axis (X)
        self._time_axis.setFormat("MM/yyyy")
        self._time_axis.setTitleText(Labels.TIME_AXIS)
        self._time_axis.setTitleFont(Fonts.BODY)
        self._time_axis.setLabelsFont(Fonts.SMALL)
        
        # State axis (Y)
        self._state_axis.setRange(-1, 1)  # Depression to Overheated
        self._state_axis.setTitleText(Labels.STATE_AXIS)
        self._state_axis.setTitleFont(Fonts.BODY)
        self._state_axis.setLabelsFont(Fonts.SMALL)
        self._state_axis.setTickCount(5)
        
        # Add axes to chart
        self._chart.addAxis(self._time_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self._state_axis, Qt.AlignmentFlag.AlignLeft)
        
        # Attach series to axes
        self._historical_series.attachAxis(self._time_axis)
        self._historical_series.attachAxis(self._state_axis)
        self._prediction_series.attachAxis(self._time_axis)
        self._prediction_series.attachAxis(self._state_axis)
        self._current_point_series.attachAxis(self._time_axis)
        self._current_point_series.attachAxis(self._state_axis)
        
    def _setup_threshold_lines(self):
        """Add threshold lines for economic states."""
        # Overheated threshold
        overheated_line = QLineSeries()
        pen = QPen(QColor(EconomicStateColors.OVERHEATED))
        pen.setWidth(Dimensions.THRESHOLD_LINE_WIDTH)
        pen.setStyle(Qt.PenStyle.DashLine)
        overheated_line.setPen(pen)
        
        # Goldilocks line (center)
        goldilocks_line = QLineSeries()
        pen = QPen(QColor(EconomicStateColors.GOLDILOCKS))
        pen.setWidth(Dimensions.THRESHOLD_LINE_WIDTH)
        pen.setStyle(Qt.PenStyle.DashLine)
        goldilocks_line.setPen(pen)
        
        # Depression threshold
        depression_line = QLineSeries()
        pen = QPen(QColor(EconomicStateColors.DEPRESSION))
        pen.setWidth(Dimensions.THRESHOLD_LINE_WIDTH)
        pen.setStyle(Qt.PenStyle.DashLine)
        depression_line.setPen(pen)
        
        # Add threshold lines
        self._chart.addSeries(overheated_line)
        self._chart.addSeries(goldilocks_line)
        self._chart.addSeries(depression_line)
        
        # Attach to axes
        overheated_line.attachAxis(self._time_axis)
        overheated_line.attachAxis(self._state_axis)
        goldilocks_line.attachAxis(self._time_axis)
        goldilocks_line.attachAxis(self._state_axis)
        depression_line.attachAxis(self._time_axis)
        depression_line.attachAxis(self._state_axis)
        
    def update_data(self, historical_data: list, predictions: list = None):
        """Update the graph with new historical data and predictions.
        
        Args:
            historical_data: List of tuples (timestamp, value)
            predictions: Optional list of tuples (timestamp, value) for future predictions
        """
        # Clear existing data
        self._historical_series.clear()
        self._prediction_series.clear()
        self._current_point_series.clear()
        
        if not historical_data:
            return
        
        # Add historical data
        for timestamp, value in historical_data:
            unix_time = time.mktime(timestamp.timetuple()) * 1000  # Convert to milliseconds
            self._historical_series.append(unix_time, value)
        
        # Add current point
        last_time, last_value = historical_data[-1]
        unix_time = time.mktime(last_time.timetuple()) * 1000
        self._current_point_series.append(unix_time, last_value)
        
        # Add predictions if available
        if predictions:
            for timestamp, value in predictions:
                unix_time = time.mktime(timestamp.timetuple()) * 1000
                self._prediction_series.append(unix_time, value)
        
        # Update axis ranges
        all_times = [t for t, _ in historical_data]
        if predictions:
            all_times.extend([t for t, _ in predictions])
            
        if all_times:
            self._time_axis.setRange(
                min(all_times),
                max(all_times)
            )
    
    def clear(self):
        """Clear all data from the graph."""
        self._historical_series.clear()
        self._prediction_series.clear()
        self._current_point_series.clear()