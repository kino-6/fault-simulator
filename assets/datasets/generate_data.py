import sys
import os
import yaml
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QSlider, QCheckBox, QPushButton
)
from pyqtgraph import mkPen, GraphicsLayoutWidget
from PyQt5.QtCore import Qt

CONFIG_FILE = "../configs/cfg.yaml"
DATASET_FILE = "simulated_dataset.csv"


class FaultMode:
    def __init__(self, name, base_freq=50):
        self.name = name
        self.enabled = False
        self.value = 0.0
        self.base_freq = base_freq
        self.checkbox = None
        self.slider = None
        self.label = None


class FaultSimulation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fault Simulation with Data Generation")
        self.setGeometry(100, 100, 1600, 900)

        # メインレイアウト
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # グラフエリア
        self.graph_layout = QHBoxLayout()
        layout.addLayout(self.graph_layout)

        # 波形プロット
        self.time_plot = GraphicsLayoutWidget()
        self.graph_layout.addWidget(self.time_plot)

        self.time_signal_plot = self.time_plot.addPlot(row=0, col=0, title="Time Domain Signal")
        self.time_signal_curve = self.time_signal_plot.plot(pen=mkPen(color="cyan", width=2))

        self.fft_signal_plot = self.time_plot.addPlot(row=1, col=0, title="Frequency Domain (FFT)")
        self.fft_signal_curve = self.fft_signal_plot.plot(pen=mkPen(color="magenta", width=2))

        # 設定エリア
        self.controls_layout = QVBoxLayout()
        layout.addLayout(self.controls_layout)

        # 故障モード
        self.modes = {
            "Misalignment": FaultMode("Misalignment"),
            "Bearing Wear": FaultMode("Bearing Wear"),
            "Resonance": FaultMode("Resonance", base_freq=200),
            "Looseness": FaultMode("Looseness"),
        }

        # 各故障モードのコントロール
        for mode in self.modes.values():
            control = self.create_fault_control(mode)
            self.controls_layout.addWidget(control)

        # Normal生成ボタンとレンジ調整
        self.normal_button = QPushButton("Generate Normal")
        self.normal_button.clicked.connect(lambda: self.generate_data(normal=True))
        self.controls_layout.addWidget(self.normal_button)

        self.normal_range = self.create_slider("Normal Range", 5, 0, 5)
        self.controls_layout.addWidget(self.normal_range["widget"])

        # Abnormal生成ボタンとレンジ調整
        self.abnormal_button = QPushButton("Generate Abnormal")
        self.abnormal_button.clicked.connect(lambda: self.generate_data(normal=False))
        self.controls_layout.addWidget(self.abnormal_button)

        self.abnormal_range = self.create_slider("Abnormal Range", 10, 5, 20)
        self.controls_layout.addWidget(self.abnormal_range["widget"])

        # コンフィグ保存ボタン
        self.save_button = QPushButton("Save Config")
        self.save_button.clicked.connect(self.save_config)
        self.controls_layout.addWidget(self.save_button)

        # コンフィグ読み込み
        self.load_config()

        # サンプルレートと時間軸
        self.sample_rate = 1000
        self.time = np.linspace(0, 1, self.sample_rate, endpoint=False)

    def create_slider(self, name, initial, minimum, maximum):
        """
        スライダーを作成
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        label = QLabel(f"{name}: {initial}%")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(initial)
        slider.valueChanged.connect(lambda value: label.setText(f"{name}: {value}%"))
        layout.addWidget(label)
        layout.addWidget(slider)
        return {"widget": widget, "slider": slider}

    def create_fault_control(self, mode):
        """
        故障モードのUIを作成
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)

        mode.checkbox = QCheckBox(f"{mode.name} Enabled")
        mode.checkbox.setChecked(mode.enabled)
        mode.checkbox.stateChanged.connect(lambda state: setattr(mode, "enabled", state == Qt.Checked))
        layout.addWidget(mode.checkbox)

        mode.label = QLabel(f"{mode.name} Value: 0")
        mode.slider = QSlider(Qt.Horizontal)
        mode.slider.setMinimum(0)
        mode.slider.setMaximum(100)
        mode.slider.setValue(0)
        mode.slider.valueChanged.connect(lambda value: self.update_fault_value(mode, value))
        layout.addWidget(mode.label)
        layout.addWidget(mode.slider)
        return widget

    def update_fault_value(self, mode, value):
        mode.value = value / 100.0
        mode.label.setText(f"{mode.name} Value: {value}")

    def generate_signal(self, normal=True, base_freq=(40, 60)):
        """
        信号を生成
        """
        base_freq = np.random.uniform(base_freq[0], base_freq[1])
        signal = np.sin(2 * np.pi * base_freq * self.time)

        range_min, range_max = (0, 0.05) if normal else (0.05, 0.2)

        enabled_modes = []
        if not normal:
            enabled_modes = np.random.choice(list(self.modes.keys()), size=np.random.randint(1, len(self.modes) + 1), replace=False)
        else:
            enabled_modes = np.random.choice(list(self.modes.keys()), size=np.random.randint(1, len(self.modes) + 1), replace=False)

        for mode_name, mode in self.modes.items():
            mode.enabled = mode_name in enabled_modes
            mode.value = np.random.uniform(range_min, range_max) if mode.enabled else 0.0

            # スライダーとスイッチに正確に反映
            mode.checkbox.setChecked(mode.enabled)
            mode.slider.setValue(int(mode.value * 100))

            if mode.enabled:
                if mode.name == "Misalignment":
                    signal += mode.value * np.sin(4 * np.pi * base_freq * self.time)
                elif mode.name == "Bearing Wear":
                    signal += mode.value * np.random.normal(size=len(self.time))
                elif mode.name == "Resonance":
                    signal += mode.value * np.sin(2 * np.pi * mode.base_freq * self.time)
                elif mode.name == "Looseness":
                    looseness = np.random.choice([0, 1], size=len(self.time), p=[0.98, 0.02])
                    signal += mode.value * looseness

        return signal

    def generate_data(self, normal=True):
        """
        データを生成し、プロットとCSV追記を行う
        """
        signal = self.generate_signal(normal)
        fft_result = np.abs(np.fft.rfft(signal))
        label = 0 if normal else 1

        # プロット更新
        self.time_signal_curve.setData(self.time, signal)
        self.fft_signal_curve.setData(np.fft.rfftfreq(len(signal), 1 / self.sample_rate), fft_result)

        # データ追記
        data = np.hstack((fft_result, [label]))
        file_exists = os.path.exists(DATASET_FILE)
        df = pd.DataFrame([data])
        with open(DATASET_FILE, mode='a') as f:
            df.to_csv(f, header=not file_exists, index=False)
        print(f"Data saved: {'Normal' if normal else 'Abnormal'}")

    def save_config(self):
        config = {
            "modes": {name: {"enabled": mode.enabled, "value": mode.value} for name, mode in self.modes.items()},
        }
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(config, f)

    def load_config(self):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f)
            for name, settings in config.get("modes", {}).items():
                mode = self.modes[name]
                mode.enabled = settings["enabled"]
                mode.value = settings["value"]
                mode.checkbox.setChecked(mode.enabled)
                mode.slider.setValue(int(mode.value * 100))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaultSimulation()
    window.show()
    sys.exit(app.exec())
