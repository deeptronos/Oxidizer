import sys
import wave
import numpy as np
from textual import events, on
from textual.app import App, ComposeResult
from textual_canvas import Canvas
from textual.widgets import Header, Footer, Button, Digits, SelectionList, OptionList
from textual.widgets.selection_list import Selection
from textual.containers import Container
from textual.color import Color


import scipy.io.wavfile as wav
from pedalboard import Plugin  # Pedalboard, Bitcrush
from pedalboard.io import AudioFile, AudioStream
import numpy as np


def SelectionPlugins():
    """A SelectionList of all plugins supplied by Pedalboard."""
    # eep = SelectionList[str]()  # Corrected to specify the type list[str]
    # for indx, val in enumerate(cls.__name__ for cls in Plugin.__subclasses__()):
    #     # eep.add_option(Selection(value=val,prompt=val[::3]))
    #     eep.add_option((val, val[::3]))
    # return eep  # Return a list of selections
    out = list()
    for cls in Plugin.__subclasses__():
        s = Selection( cls.__name__, cls.__name__[::2   ] )
        assert isinstance(s, Selection)
        out.append(s)
        
    return out

class WaveformApp(App):
    """A Textual app to display a waveform from a WAV file."""

    CSS_PATH = "waveform.tcss"

    def __init__(self, wav_file_path):
        super().__init__()
        self.wav_file_path = wav_file_path

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Digits("Catherine's 0\u03c7idizer", id="logo")
        yield Header()
        
        with Container(id="prog"):
            # yield Digits("Catherine's 0\u03c7idizer", id="logo")
            self.wf = Container(WaveformCanvas(self.wav_file_path, 15, 15))
            # yield OptionList(
            # "Aerilon",
            # "Aquaria",
            # "Canceron",
            # "Caprica",
            # "Gemenon",
            # "Leonis",
            # "Libran",
            # "Picon",
            # "Sagittaron",
            # "Scorpia",
            # "Tauron",
            # "Virgon",
            # )
            enums =SelectionList(*SelectionPlugins())
            # p=  SelectionList[int]()
            # for indx, plug in enums:
            #     p.add_option(Selection(indx,plug, False))
            # yield OptionList(*plugs)
            yield enums
            # with Container(id="plugin-list"):
            #     # for i in enumerate(enumerate_plugins):
            #     # print(f"Plugin: {i}")
            #     # yield SelectionList[str](SelectionPlugins())
            #     yield OptionList(SelectionPlugins())
            #     # for i in enumerate(enumerate_plugins):
                # print(f"Plugin: {i}")
                # self.plugins = SelectionList[str]()
                # i = 0
                # for p in enumerate_plugins():
                #     i = i + 1
                #     # self.plugins.add_options(items=[p,p,False])
                #     plugin = Selection(p, p, False)
                #     # self.plugins.add_option(Selection(p,p,False))
                #     self.plugins.add_option(plugin)

                # yield self.plugins
                # plugins = plugins.add_options(items=[Selection("DC Offset", "dc_offset", True)])
                # for plugin in enumerate_plugins():
                # plugins = plugins.add_options(items=Selection(plugin, plugin, False))
                # plugins.add_options(SelectionList[str](*enumerate_plugins()))
                # yield plugins
                # plugins = SelectionList[str](*enumerate_plugins())
                # print(f"Selections List: {plugins}")
                # plugins.select(enumerate_plugins()[0])
                # yield plugins
                # yield SelectionList[str](*enumerate_plugins())
                # yield SelectionList[str](
                #     Selection("Falken's Maze", "secret_back_door", True),
                #     Selection("Black Jack", "black_jack"),
                #     Selection("Gin Rummy", "gin_rummy"),
                #     Selection("Hearts", "hearts"),
                #     Selection("Bridge", "bridge"),
                #     Selection("Checkers", "checkers"),
                #     Selection("Chess", "a_nice_game_of_chess", True),
                #     Selection("Poker", "poker"),
                #     Selection("Fighter Combat", "fighter_combat", True),
                # )
            self.mount(self.wf)
            yield self.wf
            yield Button("Export", id="export", variant="primary")

            # yield Button('Play', id='play', variant="success"

        yield Footer()

    @on(Button.Pressed, "#export")
    def plus_minus_pressed(self) -> None:
        """Pressed Export"""
        # self.numbers = self.value = str(Decimal(self.value or "0") * -1)

    # def on_mount(self) -> None:
        # self.query_one(SelectionList).border_title = "Shall we play some games?"


class WaveformCanvas(Canvas):
    """Canvas widget for drawing the waveform."""

    def __init__(self, wav_file_path, width, height, *args, **kwargs):
        super().__init__(*args, **kwargs, width=width, height=height)
        self.wav_file_path = wav_file_path

    def on_mount(self) -> None:
        """Called when the widget is mounted. Load and draw the waveform."""
        # self.draw_grid()
        self.load_and_draw_waveform()

    def draw_grid(self):
        width, height = (15, 15)
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
