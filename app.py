import sys
import os
import yaml
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QSlider, QCheckBox, QPushButton
)
from pyqtgraph import mkPen, GraphicsLayoutWidget
from PyQt5.QtCore import Qt

CONFIG_FILE = "assets/configs/cfg.yaml"
DATASET_FILE = "simulated_results.csv"
MODEL_FILE = "assets/models/lightgbm_fault_model.txt"
SCALER_FILE = "assets/models/scaler.pkl"


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
        self.setWindowTitle("Fault Simulation with Model Prediction")
        self.setGeometry(100, 100, 1600, 900)

        # モデルとスケーラーのロード
        self.model = lgb.Booster(model_file=MODEL_FILE)
        self.scaler = joblib.load(SCALER_FILE)

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

        # 推論結果ラベル
        self.prediction_label = QLabel("Prediction: Waiting...")
        self.prediction_label.setStyleSheet("font-size: 18px; color: blue;")
        self.controls_layout.addWidget(self.prediction_label)

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

    def generate_signal(
        self,
        normal=True,
        base_freq_range=(45, 55),
        normal_range=(0, 0.05),
        fault_range=(0.05, 0.2),
        looseness_probability=(0.98, 0.02)
    ):
        """
        信号を生成

        Args:
            normal (bool): Trueの場合、正常波形を生成。それ以外の場合、異常波形を生成。
            base_freq_range (tuple): 基本周波数の範囲 (min, max)。
            normal_range (tuple): 正常波形時の成分注入範囲 (min, max)。
            fault_range (tuple): 異常波形時の成分注入範囲 (min, max)。
            looseness_probability (tuple): Loosenessモードの状態切り替え確率 (p[0]: 0, p[1]: 1)。
        """
        # 基本周波数のランダム設定
        base_freq = np.random.uniform(*base_freq_range)
        signal = np.sin(2 * np.pi * base_freq * self.time)

        # 成分注入範囲の設定
        range_min, range_max = normal_range if normal else fault_range

        # モードの有効化設定
        if normal:
            # Normalモードでは0個以上のモードがランダムに有効化される
            enabled_modes = np.random.choice(
                list(self.modes.keys()),
                size=np.random.randint(0, len(self.modes) + 1),
                replace=False
            )
        else:
            # 故障モードでは少なくとも1つのモードを有効化
            enabled_modes = np.random.choice(
                list(self.modes.keys()),
                size=np.random.randint(1, len(self.modes) + 1),
                replace=False
            )

        for mode_name, mode in self.modes.items():
            mode.enabled = mode_name in enabled_modes
            mode.value = np.random.uniform(range_min, range_max) if mode.enabled else 0.0

            # スライダーとスイッチに正確に反映
            mode.checkbox.setChecked(mode.enabled)
            mode.slider.setValue(int(mode.value * 100))

            # モードの信号成分を追加
            if mode.enabled:
                if mode.name == "Misalignment":
                    signal += mode.value * np.sin(4 * np.pi * base_freq * self.time)
                elif mode.name == "Bearing Wear":
                    signal += mode.value * np.random.normal(size=len(self.time))
                elif mode.name == "Resonance":
                    signal += mode.value * np.sin(2 * np.pi * mode.base_freq * self.time)
                elif mode.name == "Looseness":
                    looseness = np.random.choice([0, 1], size=len(self.time), p=looseness_probability)
                    signal += mode.value * looseness

        return signal

    def generate_data(self, normal=True):
        """
        データを生成し、プロット、CSV追記、推論を行う
        """
        signal = self.generate_signal(normal)
        fft_result = np.abs(np.fft.rfft(signal))
        label = 0 if normal else 1

        # 推論
        fft_normalized = self.scaler.transform([fft_result])
        prediction_prob = self.model.predict(fft_normalized)[0]
        prediction = 1 if prediction_prob > 0.5 else 0
        prediction_text = "Normal" if prediction == 0 else "Fault"

        # プロット更新
        self.time_signal_curve.setData(self.time, signal)
        self.fft_signal_curve.setData(np.fft.rfftfreq(len(signal), 1 / self.sample_rate), fft_result)

        # 推論結果表示
        self.prediction_label.setText(
            f"Prediction: {prediction_text} (Prob: {prediction_prob:.3f})"
        )
        self.prediction_label.setStyleSheet(f"font-size: 18px; color: {'green' if prediction == 0 else 'red'};")

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
