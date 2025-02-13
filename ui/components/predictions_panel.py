from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from ..utils.styling import (
    EconomicStateColors,
    Fonts,
    StyleSheets,
    Dimensions,
    Labels,
    get_severity_color
)

class PredictionBox(QFrame):
    """Individual prediction box for a specific timeframe."""
    
    def __init__(self, timeframe: str, parent=None):
        super().__init__(parent)
        self.setObjectName("predictionBox")
        self.setStyleSheet(StyleSheets.PREDICTION_BOX)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Timeframe label
        self.timeframe_label = QLabel(timeframe)
        self.timeframe_label.setFont(Fonts.SUBTITLE)
        self.timeframe_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.timeframe_label)
        
        # State prediction
        self.state_label = QLabel()
        self.state_label.setFont(Fonts.BODY)
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.state_label)
        
        # Score and confidence
        self.score_label = QLabel()
        self.score_label.setFont(Fonts.SMALL)
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_label)
        
        self.confidence_label = QLabel()
        self.confidence_label.setFont(Fonts.SMALL)
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.confidence_label)

    def update_prediction(self, state: str, score: float, confidence: float = None):
        """Update the prediction box with new values."""
        self.state_label.setText(state)
        self.state_label.setStyleSheet(f"color: {EconomicStateColors.get_state_color(score)};")
        
        self.score_label.setText(f"Score: {score:.3f}")
        
        if confidence is not None:
            self.confidence_label.setText(f"Confidence: {confidence:.1%}")
            self.confidence_label.setVisible(True)
        else:
            self.confidence_label.setVisible(False)
            
        # Update background color based on score
        bg_color = ""
        if score > 0.3:
            bg_color = EconomicStateColors.OVERHEATED_LIGHT
        elif score < -0.3:
            bg_color = EconomicStateColors.DEPRESSION_LIGHT
        else:
            bg_color = EconomicStateColors.GOLDILOCKS_LIGHT
            
        self.setStyleSheet(f"""
            QFrame#predictionBox {{
                background-color: {bg_color};
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                padding: 8px;
            }}
        """)

class PredictionsPanel(QFrame):
    """Panel showing economic predictions for multiple timeframes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("predictionsPanel")
        self.setStyleSheet(StyleSheets.PANEL)
        self.setMinimumHeight(Dimensions.PREDICTIONS_PANEL_HEIGHT)
    
        # Initialize prediction boxes first
        self.prediction_boxes = {
            '3m': None,
            '6m': None,
            '12m': None
        }
    
        # Then initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI layout and components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*Dimensions.get_panel_margins())
        
        # Title
        title_label = QLabel(Labels.PREDICTIONS)
        title_label.setFont(Fonts.TITLE)
        layout.addWidget(title_label)
        
        # Prediction boxes container
        boxes_container = QFrame()
        boxes_layout = QHBoxLayout(boxes_container)
        boxes_layout.setSpacing(Dimensions.PANEL_MARGIN)
        
        # Create prediction boxes for each timeframe
        self.prediction_boxes['3m'] = PredictionBox(Labels.THREE_MONTH)
        self.prediction_boxes['6m'] = PredictionBox(Labels.SIX_MONTH)
        self.prediction_boxes['12m'] = PredictionBox(Labels.TWELVE_MONTH)
        
        # Add boxes to layout
        for box in self.prediction_boxes.values():
            boxes_layout.addWidget(box)
            
        layout.addWidget(boxes_container)
    
    def update_predictions(self, predictions: dict):
        """Update all prediction boxes with new data.
        
        Args:
            predictions: Dictionary containing predictions for each timeframe
                Example: {
                    '3m': {'state': 'OVERHEATED', 'score': 0.8, 'confidence': 0.9},
                    '6m': {'state': 'GOLDILOCKS', 'score': 0.2, 'confidence': 0.8},
                    '12m': {'state': 'DEPRESSION', 'score': -0.5, 'confidence': 0.7}
                }
        """
        for timeframe, prediction in predictions.items():
            if timeframe in self.prediction_boxes and prediction:
                self.prediction_boxes[timeframe].update_prediction(
                    prediction['state'],
                    prediction['score'],
                    prediction.get('confidence')  # Confidence is optional
                )
    
    def clear(self):
        """Reset all prediction boxes."""
        for box in self.prediction_boxes.values():
            box.update_prediction("", 0.0)