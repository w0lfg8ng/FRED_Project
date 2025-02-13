from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QToolBar, 
                            QMenuBar, QMenu, QFileDialog, QDialog, QLabel, 
                            QSpinBox, QPushButton, QHBoxLayout, QMessageBox)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QTimer
from datetime import datetime, timedelta
import sys
from pathlib import Path
import pandas as pd

# Project imports
from data_collection import FREDCollector
from config import Config  # Changed this line
from models.state_classifier import MLStateClassifier
from models.economic_model import EconomicModel
from ui.utils.styling import StyleSheets, Dimensions, Labels
from ui.components import StatePanel, PredictionsPanel, StateGraph

class MainWindow(QMainWindow):
    """Main window for the Economic State Monitor application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(Labels.WINDOW_TITLE)
        self.setMinimumSize(Dimensions.MIN_WINDOW_WIDTH, Dimensions.MIN_WINDOW_HEIGHT)
        self.setStyleSheet(StyleSheets.WINDOW)
        
        # Initialize models
        self.classifier = MLStateClassifier()
        try:
            self.classifier.load_models('models/saved/state_classifier.joblib')
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Model not loaded")
            msg.setInformativeText("Would you like to train a new model now?")
            msg.setWindowTitle("Model Loading Error")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            retval = msg.exec()
            
            if retval == QMessageBox.StandardButton.Yes:
                self._train_new_model()
            else:
                print("Warning: Operating without a trained model. Some features may be limited.")
        
        # Initialize UI components
        self.state_panel = None
        self.predictions_panel = None
        self.state_graph = None
        
        # Set up UI
        self._init_ui()
        
        # Initialize counter before any refresh calls
        self.update_count = 0

        # Set up update timer (refresh every 5 minutes)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refresh_data)
        self.update_timer.start(300000)  # 5 minutes in milliseconds

        # Initial data load
        self.refresh_data()
   
    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        export_action = QAction("Export Data...", self)
        export_action.triggered.connect(self._export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self._show_settings)
        file_menu.addAction(settings_action)
        
    def _create_toolbar(self):
        """Create the application toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Add refresh button
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.refresh_data)
        toolbar.addAction(refresh_action)
        
        # Add train model button
        train_action = QAction("Train Model", self)
        train_action.triggered.connect(self._train_new_model)
        toolbar.addAction(train_action)
        
    def _train_new_model(self):
        """Train a new model with current data."""
        try:
            config = Config()
            collector = FREDCollector(config)
            historical_data = collector.get_historical_data()
            
            if historical_data is not None and not historical_data.empty:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setText("Training model...")
                msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
                msg.show()
                
                # Train the model
                metrics = self.classifier.train(historical_data)
                
                # Save the trained model
                self.classifier.save_models('models/saved/state_classifier.joblib')
                
                msg.close()
                
                success_msg = QMessageBox()
                success_msg.setIcon(QMessageBox.Icon.Information)
                success_msg.setText("Model trained successfully!")
                success_msg.setDetailedText(f"Training metrics:\n{str(metrics)}")
                success_msg.exec()
            else:
                QMessageBox.critical(self, "Error", "No historical data available for training.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error training model: {str(e)}")
    
    def _export_data(self):
        """Export current data to a CSV file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Data",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if filename:
                # Get current data
                current_state = self._get_current_state()
                predictions = self._get_predictions()
                historical_data = self._get_historical_data()
                
                # Create DataFrame for export
                export_data = pd.DataFrame()
                
                # Add current state
                if current_state:
                    export_data.loc[0, 'Current State'] = current_state['state'].name
                    export_data.loc[0, 'Current Score'] = current_state['score']
                    export_data.loc[0, 'Confidence'] = current_state.get('confidence', '')
                
                # Add predictions
                for timeframe, pred in predictions.items():
                    if pred:
                        export_data.loc[0, f'Prediction {timeframe} State'] = pred['state']
                        export_data.loc[0, f'Prediction {timeframe} Score'] = pred['score']
                        export_data.loc[0, f'Prediction {timeframe} Confidence'] = pred.get('confidence', '')
                
                # Add historical data
                hist_df = pd.DataFrame(historical_data, columns=['Date', 'Score'])
                
                # Save to CSV
                with pd.ExcelWriter(filename) as writer:
                    export_data.to_excel(writer, sheet_name='Current State', index=False)
                    hist_df.to_excel(writer, sheet_name='Historical Data', index=False)
                
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
    
    def _show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self.update_timer.interval(), self)
        if dialog.exec():
            # Update refresh interval
            new_interval = dialog.get_refresh_interval()
            self.update_timer.setInterval(new_interval)

    def _init_ui(self):
        """Initialize the user interface."""
        # Add menu bar and toolbar
        self._create_menu_bar()
        self._create_toolbar()
    
        # Create central widget and main layout
        central_widget = QWidget(self)  # Add 'self' as parent
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(Dimensions.PANEL_MARGIN)
        layout.setContentsMargins(*Dimensions.get_panel_margins())
    
        # Create components with explicit parents
        self.state_panel = StatePanel(central_widget)
        self.predictions_panel = PredictionsPanel(central_widget)
        self.state_graph = StateGraph(central_widget)
    
        # Add components to layout
        layout.addWidget(self.state_panel)
        layout.addWidget(self.predictions_panel)
        layout.addWidget(self.state_graph)
        
    def refresh_data(self):
        """Refresh all data from the model."""
        self.update_count += 1
        if self.update_count > 10:  # Limit to 10 updates
            self.update_timer.stop()
            return

        import traceback
    
        try:
            # Get current state
            current_state = self._get_current_state()
            if current_state:
                # Get current data
                current_data = self._get_recent_data()
            
                if current_data is None:
                    print("No current data available")
                    return
                
                # Update state panel
                print("Updating state panel...")  # Debug print
                if self.state_panel and not self.state_panel.isHidden() and self.state_panel.isVisible():
                    print(f"Current state: {current_state}")  # Debug print
                    self.state_panel.update_state(
                        state=current_state['state'].name,
                        score=current_state['score'],
                        confidence=current_state.get('confidence', None)
                    )
                
                    # Update economic metrics
                    print("Updating metrics...")  # Debug print
                    metrics = {
                        'GDP Growth': current_data['Real GDP_yoy_change'].iloc[0],
                        'Unemployment': current_data['Unemployment Rate_yoy_change'].iloc[0],
                        'Inflation': current_data['Consumer Price Index_yoy_change'].iloc[0],
                        'Industrial Production': current_data['Industrial Production_yoy_change'].iloc[0],
                        'Retail Sales': current_data['Retail Sales_yoy_change'].iloc[0],
                        'Housing Starts': current_data['Housing Starts_yoy_change'].iloc[0]
                    }
                    self.state_panel.update_metrics(metrics)
                else:
                    print("State panel is not valid!")  # Debug print
                    return
            
                # Get and update predictions
                predictions = self._get_predictions()
                if predictions and self.predictions_panel and self.predictions_panel.isVisible():
                    print("Updating predictions...")  # Debug print
                    self.predictions_panel.update_predictions(predictions)
            
                # Get and update historical data
                historical_data = self._get_historical_data()
                prediction_data = self._get_prediction_data()
                if historical_data and self.state_graph and self.state_graph.isVisible():
                    print("Updating graph...")  # Debug print
                    self.state_graph.update_data(historical_data, prediction_data)
            
        except Exception as e:
            print("Error in refresh_data:")
            traceback.print_exc()
            print(f"Error details: {str(e)}")
    
    def _get_current_state(self):
        """Get current economic state from the model."""
        try:
            # Get the most recent data point and predict
            recent_data = self._get_recent_data()
            if recent_data is not None:
                state, score, contributions = self.classifier.predict_state(recent_data)
                return {
                    'state': state,
                    'score': score,
                    'confidence': max(contributions.values()) if contributions else None
                }
        except Exception as e:
            print(f"Error getting current state: {str(e)}")
        return None
    
    def _get_predictions(self):
        """Get predictions for different time horizons."""
        predictions = {
            '3m': None,
            '6m': None,
            '12m': None
        }
        
        try:
            # Get predictions for each timeframe
            recent_data = self._get_recent_data()
            if recent_data is not None:
                for timeframe in predictions.keys():
                    # This would use your model's prediction method
                    # You'll need to implement this based on your specific model
                    state, score, confidence = self._predict_for_timeframe(recent_data, timeframe)
                    predictions[timeframe] = {
                        'state': state.name,
                        'score': score,
                        'confidence': confidence
                    }
        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            
        return predictions
    
    def _get_historical_data(self):
        """Get historical economic state data."""
        try:
            # This should return a list of (datetime, value) tuples
            # You'll need to implement this based on your data storage
            return self._load_historical_data()
        except Exception as e:
            print(f"Error getting historical data: {str(e)}")
            return []
    
    def _get_prediction_data(self):
        """Get prediction data points for graphing."""
        try:
            # This should return a list of (datetime, value) tuples for future dates
            # You'll need to implement this based on your prediction model
            return self._generate_prediction_data()
        except Exception as e:
            print(f"Error getting prediction data: {str(e)}")
            return []
    
    def _get_recent_data(self):
        """Get the most recent economic data."""
        try:
            # Initialize FRED collector
            config = Config()
            collector = FREDCollector(config)
        
            # Get latest data
            latest_data = collector.get_latest_data()
        
            if latest_data is not None and not latest_data.empty:
                return latest_data
            else:
                print("Warning: No recent data available")
                return None
            
        except Exception as e:
            print(f"Error getting recent data: {str(e)}")
            return None

    def _predict_for_timeframe(self, data, timeframe):
        """Make prediction for specific timeframe."""
        try:
            # Convert timeframe string to months
            months = int(timeframe.strip('m'))
        
            # Use the new forward prediction method
            state, score, confidence = self.classifier.predict_state_forward(data, months)
        
            return state, score, confidence
    
        except Exception as e:
            print(f"Error predicting for timeframe {timeframe}: {str(e)}")
            return None, 0.0, None

    def _load_historical_data(self):
        """Load historical economic state data."""
        try:
            # Initialize FRED collector
            config = Config()
            collector = FREDCollector(config)
        
            # Get historical data
            historical_data = collector.get_historical_data()
        
            if historical_data is None or historical_data.empty:
                return []
            
            # Process data through classifier
            results = []
            for date, row in historical_data.iterrows():
                state, score, _ = self.classifier.predict_state(row.to_frame().T)
                results.append((date, score))
            
            return sorted(results)  # Sort by date
        
        except Exception as e:
            print(f"Error loading historical data: {str(e)}")
            return []

    def _generate_prediction_data(self):
        """Generate prediction data points."""
        try:
            recent_data = self._get_recent_data()
            if recent_data is None:
                return []
            
            predictions = []
            current_date = datetime.now()
        
            # Generate predictions for next 12 months
            for months in range(1, 13):
                future_date = current_date + timedelta(days=30*months)
                state, score, _ = self._predict_for_timeframe(recent_data, f"{months}m")
                predictions.append((future_date, score))
            
            return predictions
        
        except Exception as e:
            print(f"Error generating prediction data: {str(e)}")
            return []
    
    def closeEvent(self, event):
        """Handle application shutdown."""
        self.update_timer.stop()
        event.accept()

class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""
    
    def __init__(self, current_interval, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(300, 150)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Refresh interval setting
        interval_container = QWidget()
        interval_layout = QHBoxLayout(interval_container)
        
        interval_label = QLabel("Refresh Interval (minutes):")
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(1, 60)
        self.interval_spinbox.setValue(current_interval // 60000)  # Convert ms to minutes
        
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_spinbox)
        
        # Add buttons
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        
        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        # Add widgets to main layout
        layout.addWidget(interval_container)
        layout.addWidget(button_container)
    
   