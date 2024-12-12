# Oxidizer Plugin

This is a program that can be used to apply various types of distortion and resonence to a WAV file. It accepts a WAV file for input, along with arguments specifying the desired transformation to apply to the audio. It outputs the processed audio as a WAV file. 

It is written in Python. It uses the Pedalboard library for many audio effects, and the Textual library to implement a TUI.

# Run the app
```sh
python3 -m tui_app <path_to_WAV>
```

for example:
```sh
python3 -m tui_app audio/demo/Loop2.wav
```


**Note:** this program isn't finished! There are many finishing touches I'd like to add to it. Right now, DC offset doesn't work at all, but all Plugins should do their thing. You can hear their results after enabling them by pressing "Play".
# Directory Structure, ideally...
Audio files belong in `/audio`. A selection of samples for demo purposes are available in the `/demos` subdirectory. Files output by Oxidizer will be written to the `/output` subdirectory.
# Current Directory Structure:
Source audio files live in `/audio`. A selection of samples for demo purposes are available in the `/demos` subdirectory.
The program uses `/temp/temp.wav` to store the most recent version of the OUTPUT audio.

# TODO
- Add pip install requirements
- Fix DC Offset type conversion errors... seems to be a np issue
- Finish implementing tui for controlling plugin params?
- Implement export button
- Choose a pretty color scheme