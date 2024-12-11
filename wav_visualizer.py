import sys
import wave
import numpy as np
from textual.app import App, ComposeResult
from textual_canvas import Canvas
from textual.widgets import Header, Footer
from textual.containers import Container
from textual.color import Color


import scipy.io.wavfile as wav
from pedalboard import *  # Pedalboard, Bitcrush
from pedalboard.io import AudioFile, AudioStream
import numpy as np


class WaveformApp(App):
    """A Textual app to display a waveform from a WAV file."""

    CSS_PATH = "waveform.tcss"

    def __init__(self, wav_file_path):
        super().__init__()
        self.wav_file_path = wav_file_path

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Footer()

        self.wf = Container(WaveformCanvas(self.wav_file_path, 15, 15))
        self.mount(self.wf)


class WaveformCanvas(Canvas):
    """Canvas widget for drawing the waveform."""

    def __init__(self, wav_file_path, width, height, *args, **kwargs):
        super().__init__(*args, **kwargs, width=width, height=height)
        self.wav_file_path = wav_file_path

    def on_mount(self) -> None:
        """Called when the widget is mounted. Load and draw the waveform."""
        print("!trying to load and draw waveform!")
        self.test()
        # self.draw_grid()
        self.load_and_draw_waveform()

    def draw_grid(self):
        width, height = (15, 15)
        step = 10
        for x in range(0, width, step):
            self.draw_line(x, 0, x, height, Color(255, 255, 255))
        for y in range(0, height, step):
            self.draw_line(0, y, width, y, Color(255, 255, 255))

    def test(self):
        print("test called")

    def test_file_loading(self):
        with AudioFile(self.wav_file_path) as infile:
            data = infile.read(step_size_in_samples)
            sample_rate = infile.samplerate
            num_channels = infile.num_channels

            datatype = data.dtype

            if datatype == np.int16:
                info = np.iinfo(np.int16)
            elif datatype == np.int32:
                info = np.iinfo(np.int16)
            elif datatype == np.float32:
                info = np.finfo(np.float32)
            else:
                raise ValueError(f"Unsupported data type: {datatype}")

    def load_and_draw_waveform(self):
        """Loads the WAV file and draws the waveform."""
        try:
            rate, data = wav.read(self.wav_file_path)
            datatype = data.dtype
            if datatype == np.int16:
                info = np.iinfo(np.int16)
            elif datatype == np.int32:
                info = np.iinfo(np.int16)
            elif datatype == np.float32:
                info = np.finfo(np.float32)
            else:
                raise ValueError(f"Unsupported data type: {datatype}")

            data = np.clip(data, info.min, info.max)
            data = np.frombuffer(data, dtype=datatype)

            #  Downsample for display
            width, height = (15, 15)
            downsample_factor = max(1, len(data) // width)
            downsampled_data = data[::downsample_factor]

            # Normalize to fit the canvas height
            max_val = np.max(np.abs(downsampled_data))
            if max_val > 0:
                normalized_data = (downsampled_data / (max_val)) * (height / 2)
            else:
                normalized_data = np.zeros_like(downsampled_data)

            # Remap the range from [-<half canvas height>, +<half canvas height>] to [0, <canvas height>]
            half_canvas_height = height / 2
            remapped_data = (normalized_data + half_canvas_height) * (
                height / (2 * half_canvas_height)
            )

            print(
                f"original data: {data}, downsampled: {downsampled_data}, normalized_data: {normalized_data}, remapped data: {remapped_data}"
            )
            for x in range(len(remapped_data)):
                y = remapped_data[x]
                if x == 0:
                    self.draw_line(0, int(y), 0, int(y), Color(0, 0, 255))
                else:
                    self.draw_line(
                        x - 1, int(remapped_data[x - 1]), x, int(y), Color(0, 0, 255)
                    )

        except FileNotFoundError:
            self.log(f"Error: File not found: {self.wav_file_path}")
        except wave.Error as e:
            self.log(f"Error reading WAV file: {e}")
        except Exception as e:
            self.log(f"An unexpected error occurred: {e}")


def main():
    """Entry point for the app."""
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <WAV file path>")
        sys.exit(1)

    wav_file_path = sys.argv[1]
    app = WaveformApp(wav_file_path)
    app.run()


if __name__ == "__main__":
    main()
