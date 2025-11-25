import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QMessageBox, QVBoxLayout, QHBoxLayout, QComboBox,
    QProgressBar, QTextEdit, QScrollArea
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pandas import read_csv

from models.naive_seasonal import rolling_naive_seasonal
from models.sarima import rolling_seasonal_sarima
from models.arima import rolling_seasonal_arima
from models.prophet_model import rolling_seasonal_prophet


class ModelRunner(QThread):
    """Background thread to run models without freezing GUI"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)

    def __init__(self, df, model_type, params):
        super().__init__()
        self.df = df
        self.model_type = model_type
        self.params = params

    def run(self):
        try:
            self.progress.emit(f"Running {self.model_type}...")

            if self.model_type == "NAIVE":
                results, mae, nmae, season, predicted, pct_error = rolling_naive_seasonal(
                    self.df, self.params.get('is_whole', False)
                )
            elif self.model_type == "ARIMA":
                results, mae, nmae, season, predicted, pct_error = rolling_seasonal_arima(
                    self.df,
                    self.params.get('p', 1),
                    self.params.get('d', 0),
                    self.params.get('q', 1),
                    self.params.get('is_whole', False)
                )
            elif self.model_type == "SARIMA":
                results, mae, nmae, season, predicted, pct_error = rolling_seasonal_sarima(
                    self.df,
                    1, 0, 1,
                    self.params.get('P', 1),
                    self.params.get('D', 0),
                    self.params.get('Q', 1),
                    self.params.get('S', 212),
                    self.params.get('simple_diff', 0),
                    min_train_seasons=10
                )
            elif self.model_type == "PROPHET":
                results, mae, nmae, season, predicted, pct_error = rolling_seasonal_prophet(
                    self.df, min_train_seasons=10
                )
            elif self.model_type == "GRU":
                from models.gru import rolling_seasonal_gru

                results = rolling_seasonal_gru(
                    self.df,
                    is_whole=self.params.get('is_whole', False),
                    seq_len=self.params.get('seq_len', 120),
                    hidden_dim=self.params.get('hidden_dim', 128),
                    epochs=self.params.get('epochs', 200),
                    patience=self.params.get('patience', 15),
                    min_train_seasons=10
                )
                # GRU returns different format
                self.finished.emit({'type': 'gru', 'results': results})
                return

            self.finished.emit({
                'type': 'standard',
                'mae': mae,
                'nmae': nmae,
                'season_mean': season,
                'predicted': predicted,
                'pct_error': pct_error,
                'results': results
            })
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")


class ResultsWindow(QWidget):
    """Window to display forecast results"""

    def __init__(self, station_name, results):
        super().__init__()
        self.setWindowTitle(f"Results - {station_name}")
        self.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout()

        text = QTextEdit()
        text.setReadOnly(True)

        if results['type'] == 'standard':
            text.append(f"=== {station_name} Forecast Results ===\n")
            text.append(f"MAE: {results['mae']:.3f}")
            text.append(f"NMAE: {results['nmae']:.3f}")
            text.append(f"Season Mean: {results['season_mean']:.3f} cm")
            text.append(f"Predicted: {results['predicted']:.3f} cm")
            text.append(f"Error: {results['pct_error']:.3f}%")
            text.append(f"\n{results['results'].tail(10).to_string()}")
        else:
            text.append(f"=== {station_name} GRU Results ===\n")
            text.append(str(results['results']))

        layout.addWidget(text)
        self.setLayout(layout)


class DraggableScrollArea(QScrollArea):
    """Scroll area with click-and-drag panning"""

    def __init__(self, coord_label):
        super().__init__()
        self.setWidgetResizable(False)
        self._dragging = False
        self._drag_start = None
        self.coord_label = coord_label
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Update coordinates
        if self.widget():
            widget_pos = self.widget().mapFromParent(event.position().toPoint())
            self.coord_label.setText(f"X: {widget_pos.x()}, Y: {widget_pos.y()}")

        if self._dragging:
            delta = event.position().toPoint() - self._drag_start
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self._drag_start = event.position().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)



class AlpsGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snow Depth Forecaster - Alps Stations")
        self.setGeometry(100, 100, 1000, 700)

        self.station_coords = {
            'serre_chevalier_daily': (920, 1023),
            "tignes_daily": (1081, 721),
            "les2alpes_daily": (733, 984),
            "col_de_porte_daily": (552, 832),
        }

        self.coord_label = QLabel("X: 0, Y: 0")
        self.coord_label.setStyleSheet("font-family: monospace; padding: 5px;"
                                       "background-color: rgba(255, 255, 255, 128);")


        self.load_data()

        main_layout = QVBoxLayout()

        main_layout.addWidget(self.coord_label)

        content_layout = QHBoxLayout()

        map_widget = self.create_map_widget()
        content_layout.addWidget(map_widget, stretch=2)

        control_widget = self.create_control_widget()
        content_layout.addWidget(control_widget, stretch=1)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

        self.current_station = None
        self.model_thread = None

    def create_map_widget(self):
        """Create scrollable map with clickable station buttons"""

        scroll = DraggableScrollArea(self.coord_label)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container widget for map and buttons
        widget = QWidget()
        widget.setFixedSize(1499, 1856)
        widget.setMouseTracking(True)

        # Map background
        map_label = QLabel(widget)
        map_label.setMouseTracking(True)
        if Path("alps_map_crop.png").exists():
            pix = QPixmap("alps_map_crop.png")
            map_label.setPixmap(pix)
            map_label.setFixedSize(1499, 1856)
        else:
            map_label.setText("Map not found")
            map_label.setStyleSheet("background-color: #E8F4F8; border: 2px solid #999;")
            map_label.setFixedSize(1499, 1856)

        # Add station buttons at specified coordinates
        for station_name in self.datasets.keys():
            if station_name in self.station_coords:
                x, y = self.station_coords[station_name]
                btn = QPushButton(station_name, widget)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 15px;
                        padding: 5px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                btn.move(x, y)
                btn.clicked.connect(lambda checked, s=station_name: self.select_station(s))

        # Put widget inside scroll area
        scroll.setWidget(widget)
        return scroll

    def load_data(self):
        """Load all cleaned data files"""
        self.datasets = {}
        folder = Path("./cleaned v2")

        if not folder.exists():
            QMessageBox.warning(self, "Warning", "Cleaned data folder not found!")
            return

        for file in folder.iterdir():
            if file.is_file():
                self.datasets[file.stem] = read_csv(file)

    def create_control_widget(self):
        """Create control panel with model selection"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Station label
        self.station_label = QLabel("No station selected")
        self.station_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.station_label)

        # Model selection
        layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["NAIVE", "ARIMA", "SARIMA", "PROPHET", "GRU"])
        layout.addWidget(self.model_combo)

        # Run button
        self.run_btn = QPushButton("Run Forecast")
        self.run_btn.clicked.connect(self.run_forecast)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()
        widget.setLayout(layout)

        return widget

    def select_station(self, station_name):
        """Handle station selection"""
        self.current_station = station_name
        self.station_label.setText(f"Selected: {station_name}")
        self.run_btn.setEnabled(True)
        self.status_label.setText("")

    def run_forecast(self):
        """Run the selected model on the selected station"""
        if not self.current_station:
            return

        model_type = self.model_combo.currentText()
        df = self.datasets[self.current_station]

        # Model parameters (you can add UI controls for these)
        params = {
            'is_whole': 'whole' in self.current_station.lower(),
            'p': 1, 'd': 0, 'q': 1,
            'P': 1, 'D': 0, 'Q': 1, 'S': 212, 'simple_diff': 0,
            'seq_len': 120, 'hidden_dim': 128, 'epochs': 200, 'patience': 15
        }

        # Disable button and show progress
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Run in background thread
        self.model_thread = ModelRunner(df, model_type, params)
        self.model_thread.progress.connect(self.update_status)
        self.model_thread.finished.connect(self.show_results)
        self.model_thread.start()

    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)

    def show_results(self, results):
        """Display forecast results"""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Complete!")

        # Open results window
        self.results_window = ResultsWindow(self.current_station, results)
        self.results_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AlpsGUI()
    gui.show()
    sys.exit(app.exec())
