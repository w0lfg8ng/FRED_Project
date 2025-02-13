import sys
from pathlib import Path
import os

# Add the project root to Python path first
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Then import the models and other modules
from models.state_classifier import MLStateClassifier
from models.economic_model import EconomicModel
from PyQt6.QtWidgets import QApplication
from ui.main import MainWindow

def main():
    """Initialize and run the Economic State Monitor application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()