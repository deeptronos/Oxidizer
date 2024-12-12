import sys
import wave
import numpy as np
from math import floor
from textual import events, on
from textual.app import App, ComposeResult
from textual_canvas import Canvas
from textual.widget import Widget
from textual.widgets import Header, Footer, Button, Digits, SelectionList, OptionList, Input, Label
from textual.widgets.selection_list import Selection
from textual.message import Message
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.color import Color
from textual.events import Mount


import importlib
import sounddevice as sd
import scipy.io.wavfile as wav
from pedalboard import * #Plugin  # Pedalboard, Bitcrush # TODO explicitly import used Plugins to avoid syntax errors
from pedalboard.io import AudioFile, AudioStream
import numpy as np

# Credit to https://stackoverflow.com/questions/12262463/print-out-the-class-parameters-on-python
def filterFunction(field):
    '''
    This is a sample filtering function
    '''
    name, value = field
    return not name.startswith("_")


def get_all_methods_details(class_name):
    out = []
    for i,plugin in enumerate(filter(filterFunction, class_name.__dict__.items())):
        print(i, plugin)
        out.append((i, plugin))
    return out


def SelectionPlugins():
    """A list of all Pedalboard plugins supplied to the user."""
    out = list()
    banned_plugins=["PluginContainer", "Convolution", "Resample", "IIRFilter", "PrimeWithSilenceTestPlugin", "AddLatency", "ResampleWithLatency", "FixedSizeBlockTestPlugin", "ForceMonoTestPlugin", "ExternalPlugin"]
    for cls in Plugin.__subclasses__():
        # s = Selection(cls.__name__, cls.__name__[::2])
        if cls.__name__ in banned_plugins:
            continue
        s = Selection(cls.__name__, cls.__name__)
        assert isinstance(s, Selection)
        out.append(s)

    return out


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
            
        except FileNotFoundError:
            print(f"TempStore - Error: File not found: {self._originalpath}")
        except wave.Error as e:
            print(f"TempStore - Error reading WAV file: {e}")
        except Exception as e:
            print(f"TempStore - An unexpected error occurred: {e}")
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
        self._write_to_temp(data)
    
    def read(self):
        """Returns a tuple containing the (rate, data, info, temppath) read from temp store. Temppath is the path of the file in the TempStore()."""
        out_tuple = self._read_from_temp()
        out = list(out_tuple)+list([self._temppath])
        return tuple(out)


class PluginInterfaceInput(Input):
    """Responsible for a single input field in an interactable plugin."""
    class InputChanged(Message):
        """Input has been changed message."""
        def __init__(self, paramname:str, value:int) -> None:
            self.value = value
            self.paramname =paramname
            super.__init__()

    def __init__(self, placeholder, type):
        super().__init__(placeholder=placeholder,type=type)
    
    def on_changed(self, event: Input.Changed ):
        self.post_message(InputChanged(paramname=event.name,value=int(event.value)))


class PluginWidget(Widget):
    """Responsible for a single user-interactable plugin."""

    def __init__(self, plugin:Plugin):
        self._plugin : Plugin = plugin # TODO valid for Plugin subclasses?
        self._pluginfo = get_all_methods_details(plugin)

    def compose(self):
        super().compose()
        # return super().compose()
        for i, param in get_all_methods_details(self._plugin):
            name = param[0]
            prop = param[1]
            # yield Input(placeholder=name, type="integer")
            yield PluginInterfaceInput(placeholder=name,type="integer")
        yield
    def on_mount(self) -> None:
        self.border_title = self._plugin

    def handle_input_changed(self, message:PluginInterfaceInput.InputChanged):
        setattr(self._plugin, message.paramname, message.value)


class AudioFX():
    """Responsible for the FX applied to audio.
Writes processed audio to the app's TempStore.
FX are applied to audio in the order in which they are added to the Pedalboard.

    """
    def __init__(self, source_file_path:str, temp_store:TempStore):
        self._source_file_path = source_file_path
        self._pedalboard = Pedalboard()
        self._ts = temp_store
        self._load_source_audio()
        self._dc_offset = 0

    def _load_source_audio(self):
        """Read data from the file at _source_file_path into _data"""
        (self._rate, self._source_data, self._info, self._temppath) = self._ts.read()
        
    def _apply_fx_to_source(self):
        """Store the processed audio in _data into the TempStore."""
        print(f"apply fx to source. fx:")
        for i in list(self._pedalboard):
            print(f"\t{i}")
        # self._temppath
        with AudioFile(self._source_file_path) as infile:
            with AudioFile(self._temppath, 'w', infile.samplerate, infile.num_channels) as outfile:
                while infile.tell() < infile.frames:
                    chunk = infile.read(infile.samplerate * infile.duration) # TODO this is not so good for memory ;3
            
                effected = self._pedalboard(chunk, infile.samplerate, reset=True)

                outfile.write(effected)

    def _update_dc_offset(self):
        """Update the DC Offset of the audio signal."""
        (rate, source_data, info, temppath) = self._ts.read()
        print(f"self._dc_offset: {self._dc_offset}")
        modified_data = source_data + (int(self._dc_offset).astype(source_data.dtype))
        
        modified_data = np.clip(modified_data, info.min, info.max)
        self._ts.store(modified_data)

    def enable_plugin(self, plugin_name:str):
        """Add the given plugin to the end of the Pedalboard Effects chain."""
        print(f"enable_plugin: {plugin_name}")
        module = importlib.import_module("pedalboard")
        class_ = getattr(module, plugin_name)
        if(class_().__class__.__name__ not in [activeplugin.__class__.__name__ for activeplugin in list(self._pedalboard)]):

            self._pedalboard.append(class_())
            self._apply_fx_to_source()
        else:
            pass

    def disable_plugin(self, plugin_name:str):
        """Remove the given plugin from the Pedalboard Effects chain."""
        print(f"disable_plugin: {plugin_name}")
        module = importlib.import_module("pedalboard")
        class_ = getattr(module, plugin_name)
        if(class_().__class__.__name__ in [activeplugin.__class__.__name__ for activeplugin in list(self._pedalboard)]):
            for i in range(0, len(self._pedalboard)):
                if self._pedalboard[i].__class__.__name__ == class_().__class__.__name__:
                    del self._pedalboard[i]
                    break
            # self._pedalboard.remove(class_().__class__.__name__)
            self._apply_fx_to_source()
        else:
            pass

    def set_dc_offset(self, offset):
        self._dc_offset = offset
        self._update_dc_offset()

    def get_dc_offset(self):
        return self._dc_offset
    
    def get_pedalboard(self) -> Pedalboard:
        return self._pedalboard


class PluginContainer(ScrollableContainer):
    """Responsible for a widget displaying active audio FX and their inputs.
    """
    def compose(self):
        for i in list(self._fx.get_pedalboard()):
            yield PluginWidget(i)
        super().compose()

    def set_fx_source(self, audio_fx:AudioFX):
        """Set self._fx"""
        self._fx = audio_fx

    def _get_plugins(self):
        """Get a list enumerating the active plugins in our pedalboard
        """
        return list(enumerate(self._fx.get_pedalboard()))
    


class WaveformApp(App):
    """A Textual app to display a waveform from a WAV file."""

    CSS_PATH = "waveform.tcss"

    class DCInputChanged(Message):
        """Input has been changed message."""
        def __init__(self, value:int) -> None:
            self.value = value
            super.__init__()

    def __init__(self, wav_file_path):
        super().__init__()
        self.wav_file_path = wav_file_path
        self.ts = TempStore(self.wav_file_path) # The TempStore() backing the output waveform.
        self.audio_device_name = sd.query_devices()[len(sd.query_devices()) - 1 ]['name'] # TODO better solution than just picking the last one...

        self.fx = AudioFX(self.wav_file_path, self.ts)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Digits("0\u03c7idizer", id="logo")
        yield Header()

        with Container(id="grid-prog"):
            # yield Digits("Catherine's 0\u03c7idizer", id="logo")
            self.source_wf = WaveformCanvas(self.wav_file_path,"SOURCE", 20, 20)
            self.source_canvas = Container(self.source_wf, classes="wf", id="source-wf")
            (temprate, tempdata, tempinfo, temppath) = self.ts.read()
            self.modified_wf = WaveformCanvas(temppath, "OUTPUT", 20, 20)
            self.modified_canvas = Container(self.modified_wf, classes="wf", id="modified-wf")
            

            # self.mount(self.source_wf)
            # self.mount(self.modified_wf)
            yield self.source_canvas
            yield self.modified_canvas
            yield Container(SelectionList(*SelectionPlugins()), id="plugin-selection-list")
            self._plugin_interactables = PluginContainer()
            self._plugin_interactables.set_fx_source(self.fx)
            # with Container(id="dc-offset"):
            #     yield Label("DC OFFSET:")
            #     yield Input(placeholder="0", type="integer")
            dc_input = Input(placeholder="0", type="integer", id="dc-offset-input")
            dc_input.border_title = "DC Offset"
            dc_input.border_subtitle = "Press <enter> to submit."
            yield Horizontal(
                dc_input
            )
            self._interactions = Container(self._plugin_interactables,id="plugin-interactions")
           
            yield Horizontal(
                Button("Play", id="play", variant="success"),
                Button("Export", id="export", variant="primary"),
            )

        yield Footer()
        

    @on(Input.Changed, "#dc-offset-input")
    def on_changed(self, event: Input.Changed ):
        # self.post_message(InputChanged(value=int(event.value)))
        self.log("DC Offset changed to: ", event.value)
        self.fx.set_dc_offset(event.value)

    @on(Button.Pressed, "#export")
    def export_pressed(self) -> None:
        """Pressed Export"""

    @on(Button.Pressed, "#play")
    def play_pressed(self) -> None:
        """Pressed Play"""
        self.log("Play pressed!")
        (rate, data, info, temppath) = self.ts.read()
        with AudioFile(temppath) as f:
            chunk = f.read(f.samplerate * f.duration)
        AudioStream.play(chunk, f.samplerate, self.audio_device_name)

    # @on(Mount)
    @on(SelectionList.SelectedChanged)
    def update_selected_plugins(self, msg: SelectionList.SelectedChanged) -> None:
        for i in msg.selection_list.selected:
            self.fx.enable_plugin(i)
            # self.fx.disable_plugin
        disabled_plugins = [plug for plug in list(self.fx.get_pedalboard()) if plug.__class__.__name__ not in msg.selection_list.selected]
        for plug in disabled_plugins:
            self.fx.disable_plugin(plug.__class__.__name__) # TODO this, combined with the disable_plugin logic, is ridiculous
        # self.log(f"msg.selection_list: {enumerate(msg.selection_list)}")
        # self.fx.enable_plugin

    def on_mount(self) -> None:
        self.query_one(SelectionList).border_title = "Audio Effects"





class WaveformCanvas(Canvas):
    """Canvas widget for drawing the waveform."""

    def __init__(self, wav_file_path, window_title, width, height, *args, **kwargs):
        super().__init__(*args, **kwargs, width=width, height=height)
        self.wav_file_path = wav_file_path
        self.window_title = window_title
    
    def on_mount(self) -> None:
        """Called when the widget is mounted. Load and draw the waveform."""
        # self.draw_grid()
        self.load_and_draw_waveform()
        self.border_subtitle = self.wav_file_path
        self.border_title = self.window_title
    # def compose(self) -> None:
        

    def on_show(self) -> None: # TODO causes error when program quits?
        
        self.update_size()

    def update_size(self):
        source_handle = self.parent
        width, height = (source_handle.container_size.width, source_handle.container_size.height)
        if width > 0 and height > 0:
            source_handle.remove_children("WaveformCanvas")
            twf = WaveformCanvas(self.wav_file_path, self.window_title, width, floor(height * 2))
            # twf.border_subtitle = self.border_subtitle
            
            source_handle.mount(twf)

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
