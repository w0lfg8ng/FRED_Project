from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from enum import Enum
from ..utils.styling import (
    EconomicStateColors, 
    Fonts, 
    StyleSheets, 
    Dimensions,
    Labels
)

class StatePanel(QFrame):
    """Panel showing current economic state and confidence score."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statePanel")
        self.setStyleSheet(StyleSheets.STATE_PANEL)
        self.setMinimumHeight(Dimensions.STATE_PANEL_HEIGHT)
        
        # Initialize UI
        self._init_ui()
        
        # Set default state
        self._current_state = None
        self._current_score = None
        self._confidence = None
    
    def _init_ui(self):
        """Initialize the UI layout and components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*Dimensions.get_panel_margins())
    
        # Title
        title_label = QLabel(Labels.CURRENT_STATE, parent=self)  # Add parent
        title_label.setFont(Fonts.TITLE)
        title_label.setStyleSheet("color: #000000;")  # Ensure black text
        layout.addWidget(title_label)
    
        # State container with white background
        state_container = QFrame(parent=self)  # Add parent
        state_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        state_layout = QHBoxLayout(state_container)
    
        # Create left side for state
        left_container = QFrame(parent=state_container)  # Add parent
        left_layout = QVBoxLayout(left_container)
    
        # State label (will be updated)
        self._state_label = QLabel(parent=left_container)  # Add parent
        self._state_label.setFont(Fonts.TITLE)
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self._state_label)
    
        state_layout.addWidget(left_container)
        state_layout.addStretch()
    
        # Create right side for score and confidence
        right_container = QFrame(parent=state_container)  # Add parent
        right_layout = QVBoxLayout(right_container)
    
        # Score label (will be updated)
        self._score_label = QLabel(parent=right_container)  # Add parent
        self._score_label.setFont(Fonts.BODY)
        self._score_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(self._score_label)
    
        # Confidence label (will be updated)
        self._confidence_label = QLabel(parent=right_container)  # Add parent
        self._confidence_label.setFont(Fonts.SMALL)
        self._confidence_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(self._confidence_label)
    
        state_layout.addWidget(right_container)
        layout.addWidget(state_container)
    
        # Add metrics section
        metrics_container = QFrame(parent=self)  # Add parent
        metrics_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                margin-top: 10px;
            }
        """)
        metrics_layout = QHBoxLayout(metrics_container)
        metrics_layout.setSpacing(20)
    
        # Initialize metric widgets dictionary
        self._metric_widgets = {}
    
        # Define metrics to display
        metrics = [
            ('GDP Growth', '%'),
            ('Unemployment', '%'),
            ('Inflation', '%'),
            ('Industrial Production', '%'),
            ('Retail Sales', '%'),
            ('Housing Starts', '%')
        ]
    
        # Create metric widgets
        for metric_name, unit in metrics:
            metric_frame = QFrame(parent=metrics_container)  # Add parent
            metric_frame.setStyleSheet("""
                QFrame {
                    border: 1px solid #e0e0e0;
                    border-radius: 4px;
                    padding: 8px;
                    background-color: white;
                }
            """)
            metric_layout = QVBoxLayout(metric_frame)
            metric_layout.setSpacing(4)
        
            # Metric title
            title = QLabel(metric_name, parent=metric_frame)  # Add parent
            title.setFont(Fonts.SMALL)
            title.setStyleSheet("color: #666666;")  # Dark gray for titles
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            metric_layout.addWidget(title)
        
            # Metric value
            value = QLabel('--', parent=metric_frame)  # Add parent
            value.setFont(Fonts.BODY)
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            metric_layout.addWidget(value)
        
            # Store widget reference
            self._metric_widgets[metric_name] = value
            metrics_layout.addWidget(metric_frame)
    
        layout.addWidget(metrics_container)
    
    def update_state(self, state: str, score: float, confidence: float = None):
        """Update the displayed economic state and score."""
        self._current_state = state
        self._current_score = score
        self._confidence = confidence
        
        # Update state label with appropriate color and text
        self._state_label.setText(state)
        if state == "OVERHEATED":
            color = EconomicStateColors.OVERHEATED
            bg_color = EconomicStateColors.OVERHEATED_LIGHT
        elif state == "DEPRESSION":
            color = EconomicStateColors.DEPRESSION
            bg_color = EconomicStateColors.DEPRESSION_LIGHT
        else:
            color = EconomicStateColors.GOLDILOCKS
            bg_color = EconomicStateColors.GOLDILOCKS_LIGHT
            
        self._state_label.setStyleSheet(f"color: {color};")
        
        # Update score label
        self._score_label.setText(f"Score: {score:.3f}")
        self._score_label.setStyleSheet("color: #000000;")
        
        # Update confidence if provided
        if confidence is not None:
            self._confidence_label.setText(f"Confidence: {confidence:.1%}")
            self._confidence_label.setStyleSheet("color: #000000;")
            self._confidence_label.setVisible(True)
        else:
            self._confidence_label.setVisible(False)
            
        # Update panel background
        self.setStyleSheet(f"""
            QFrame#statePanel {{
                background-color: {bg_color};
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                padding: 10px;
            }}
        """)
    
    def get_current_state(self):
        """Return current state information."""
        return {
            'state': self._current_state,
            'score': self._current_score,
            'confidence': self._confidence
        }

    def clear(self):
        """Reset the panel to empty state."""
        self._state_label.setText("")
        self._score_label.setText("")
        self._confidence_label.setText("")
        self._current_state = None
        self._current_score = None
        self._confidence = None
        
        # Clear metrics as well
        for widget in self._metric_widgets.values():
            widget.setText("--")
            widget.setStyleSheet("")

    def update_metrics(self, metrics: dict):
        """Update the economic metrics display.
        
        Args:
            metrics: Dictionary of metric values {'GDP Growth': 2.5, ...}
        """
        for metric_name, value in metrics.items():
            if metric_name in self._metric_widgets:
                widget = self._metric_widgets[metric_name]
                formatted_value = f"{value:+.1f}%"
                widget.setText(formatted_value)
                
                # Set color based on value using new metric colors
                if value > 0:
                    color = EconomicStateColors.METRIC_POSITIVE
                elif value < 0:
                    color = EconomicStateColors.METRIC_NEGATIVE
                else:
                    color = EconomicStateColors.METRIC_NEUTRAL
                    
                widget.setStyleSheet(f"color: {color};")