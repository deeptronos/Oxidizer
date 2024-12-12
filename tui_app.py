import sys
import wave
import numpy as np
from math import floor
from textual import events, on
from textual.app import App, ComposeResult
from textual_canvas import Canvas
from textual.widgets import Header, Footer, Button, Digits, SelectionList, OptionList
from textual.widgets.selection_list import Selection
from textual.containers import Container, Horizontal, Vertical
from textual.color import Color


import scipy.io.wavfile as wav
from pedalboard import Plugin  # Pedalboard, Bitcrush
from pedalboard.io import AudioFile, AudioStream
import numpy as np

class TempStore():
    """A class that can store one WAV file in `/temp`, for manipulation."""
    def __init__(self, original_file):
        self._temppath = "temp/temp.wav" # The path where temps will be written/read from.
        self._originalpath  = original_file # The path of the original that we're storing temporaries of.
        try:
            rate, data = wav.read(self._originalpath)
            datatype = data.dtype
            if datatype == np.int16:
                info = np.iinfo(np.int16)
            elif datatype == np.int32:
                info = np.iinfo(np.int16)
            elif datatype == np.float32:
                info = np.finfo(np.float32)
            else:
                raise ValueError(f"Unsupported data type: {datatype}")

            self._rate = rate # Samplerate
            self._data = data # Original sample's data
            self._info = info # Data type info
            print("Writing temp wav...")
            
        except FileNotFoundError:
            print(f"Error: File not found: {self._originalpath}")
        except wave.Error as e:
            print(f"Error reading WAV file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        wav.write(self._temppath, self._rate, self._data)


    def _get_data_dtype(self, data):
        """Get the data type of data."""
        datatype = data.dtype
        if datatype == np.int16:
            info = np.iinfo(np.int16)
        elif datatype == np.int32:
            info = np.iinfo(np.int16)
        elif datatype == np.float32:
            info = np.finfo(np.float32)
        else:
            raise ValueError(f"Unsupported data type: {datatype}")
        
        return info



    def _write_to_temp(self, data):
        """Writes the given data to a temp WAV file of rate self._rate and info self._info."""
        self._info = self._get_data_dtype(data)
        data = np.clip(data, self._info.min, self._info.max)  # Ensure data is within valid range
        wav.write(self._temppath, self._rate, data.astype(self._info.dtype))
    
    def _read_from_temp(self):
        """Returns a tuple containing the (rate, data, info) read from temp store."""
        try:
            rate, data = wav.read(self._temppath)
            datatype = data.dtype
            if datatype == np.int16:
                info = np.iinfo(np.int16)
            elif datatype == np.int32:
                info = np.iinfo(np.int16)
            elif datatype == np.float32:
                info = np.finfo(np.float32)
            else:
                raise ValueError(f"Unsupported data type: {datatype}")

            self._rate = rate # Samplerate
            self._data = data # Original sample's data
            self._info = info # Data type info
            return(self._rate, self._data, self._info)
        except FileNotFoundError:
            print(f"Error: File not found: {self._temppath}")
        except wave.Error as e:
            print(f"Error reading WAV file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def store(self, data):
        """Store data in a temp WAV. The stored WAV will have the rate and info.dtype of the sample initially passed during construction of TempStore."""
        self._data = data
        self._write_to_temp(self.data)
    
    def read(self):
        """Returns a tuple containing the (rate, data, info, temppath) read from temp store. Temppath is the path of the file in the TempStore()."""
        out_tuple = self._read_from_temp()
        out = list(out_tuple)+list([self._temppath])
        return tuple(out)


def SelectionPlugins():
    """A list of all plugins supplied to the user. Plugins originate from Pedalboard as well as myself."""
    out = list()
    for cls in Plugin.__subclasses__():
        s = Selection(cls.__name__, cls.__name__[::2])
        assert isinstance(s, Selection)
        out.append(s)

    return out


class WaveformApp(App):
    """A Textual app to display a waveform from a WAV file."""

    CSS_PATH = "waveform.tcss"

    def __init__(self, wav_file_path):
        super().__init__()
        self.wav_file_path = wav_file_path
        self.ts = TempStore(self.wav_file_path) # The TempStore() backing the output waveform.

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Digits("Catherine's 0\u03c7idizer", id="logo")
        yield Header()

        with Container(id="grid-prog"):
            # yield Digits("Catherine's 0\u03c7idizer", id="logo")
            self.source_wf = WaveformCanvas(self.wav_file_path, 20, 20)
            self.source_canvas = Container(self.source_wf, classes="wf", id="source-wf")
            (temprate, tempdata, tempinfo, temppath) = self.ts.read()
            self.modified_wf = WaveformCanvas(temppath, 20, 20)
            self.modified_canvas = Container(self.modified_wf, classes="wf", id="modified-wf")
            

            # self.mount(self.source_wf)
            # self.mount(self.modified_wf)
            yield self.source_canvas
            yield self.modified_canvas
            yield Container(SelectionList(*SelectionPlugins()), id="plugins")
            yield Horizontal(
                Button("Export", id="export", variant="primary"),
                Button("Play", id="play", variant="success"),
            )

        yield Footer()
        

    @on(Button.Pressed, "#export")
    def plus_minus_pressed(self) -> None:
        """Pressed Export"""
        # self.numbers = self.value = str(Decimal(self.value or "0") * -1)
        self.update_canvas_size()

    def on_mount(self) -> None:
        self.query("#source-wf")[0].query_one("WaveformCanvas").border_title = "SOURCE"
        self.query("#modified-wf")[0].query_one("WaveformCanvas").border_title = "RESULT"
        (_, _, _, modified_subtitle) = self.ts.read()
        self.query("#source-wf")[0].query_one("WaveformCanvas").border_subtitle= f"{self.wav_file_path}"
        self.query("#modified-wf")[0].query_one("WaveformCanvas").border_subtitle= f"{modified_subtitle}"

        self.query_one(SelectionList).border_title = "Audio Effects "
        
    def on_size(self):
        self.update_canvas_size

    def update_canvas_size(self):
        for i in ["#source-wf", "#modified-wf"]:
            #     container = self.query(i)[0]
            #     width = container.size.width
            #     height = container.size.height
            source_handle =  self.query(i)[0]
            # viewport = self.screen.size
            width, height = (source_handle.container_size.width, source_handle.container_size.height)
            if width > 0 and height > 0:
                source_handle.remove_children("WaveformCanvas")
                twf = WaveformCanvas(self.wav_file_path, width, floor(height * 2))
                source_handle.mount(twf)
                # self.log("h")
            # self.log(f"Source handle: size: {source_handle.size}, container size: {source_handle.container_size}, container viewport: {source_handle.container_viewport}, virtual_region: {source_handle.virtual_region},  with margin: {source_handle.virtual_region_with_margin}, content size: {source_handle.content_size}, virtual size: {source_handle.virtual_size}, outer_size: {source_handle.outer_size}, get_content_width: {source_handle.get_content_width(source_handle.size,viewport)}")
            # source_handle.mount(WaveformCanvas(self.wav_file_path, source_handle.size.width, source_handle.size.height * 2))
            #     self.log(f"{i}: width: {width}, height: {height}")


class WaveformCanvas(Canvas):
    """Canvas widget for drawing the waveform."""

    def __init__(self, wav_file_path, width, height, *args, **kwargs):
        print("WaveformCanvas: width: {width}, height: {height}")
        super().__init__(*args, **kwargs, width=width, height=height)
        self.wav_file_path = wav_file_path

    def on_mount(self) -> None:
        """Called when the widget is mounted. Load and draw the waveform."""
        # self.draw_grid()
        self.load_and_draw_waveform()

    def _on_click(self, event):
        self.log("On click!!!")
        self.update_size()
        return super()._on_click(event)

    def update_size(self):
        if self.parent:
            parent = self.parent
            self.log(f"Update size!!!!!! self.parent.size: {self.parent.size}")
            self.log("Add child...")
            parent.compose_add_child(WaveformCanvas(self.wav_file_path, parent.size.width, parent.size.height))
            # self.remove()
            
            

    def draw_grid(self):
        width, height = (self.width, self.height)
        step = 10
        for x in range(0, width, step):
            self.draw_line(x, 0, x, height, Color(255, 255, 255))
        for y in range(0, height, step):
            self.draw_line(0, y, width, y, Color(255, 255, 255))

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
            width, height = (self.width, self.height)
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
