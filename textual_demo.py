from textual.app import App, ComposeResult
from textual.widgets import Sparkline

from textual_canvas import Canvas

import numpy as np
from textual.app import App, ComposeResult
from textual.color import Color
from textual_canvas import Canvas
from textual.widgets import Header, Footer, Static


class WaveFormApp(App):
    """A Textual app to visualize waveform."""
    BINDINGS= [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Canvas(500,50)

    def on_mount(self) -> None:
        """Draw when the app starts."""
        self.draw_waveform()
    
    def draw_waveform(self) -> None:
        """Draw a simple waveform."""
        canvas = self.query_one(Canvas)
        width, height = canvas.size

        # Generate sine wave
        x_values = np.linspace(0, 4 * np.pi, width) # TODO: adjust stop...
        y_values = np.sin(x_values)

        y_values = (y_values + 1) / 2 # Scale to [0, 1]
        y_values *= height

if __name__ == "__main__":
    app = WaveFormApp()
    app.run()



# data = [1, 2, 2, 1, 1, 4, 3, 1, 1, 8, 8, 2]


# class SparklineBasicApp(App[None]):
#     CSS_PATH = "sparkline_basic.tcss"

#     def compose(self) -> ComposeResult:
#         yield Sparkline(
#             data,
#             summary_function=max,
#         )


# app = SparklineBasicApp()
# if __name__ == "__main__":
#     app.run()
