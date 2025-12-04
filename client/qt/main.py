#!/usr/bin/env python3
"""
Live Talker Qt Client - Main Entry Point
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Configure logging
from utils.logger import setup_logger
logger = setup_logger(level="INFO")

from main_window import MainWindow


def main():
    """Main function"""
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Live Talker")
    
    # Note: High DPI support is enabled by default in PyQt6
    # No need to set AA_EnableHighDpiScaling or AA_UseHighDpiPixmaps
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

