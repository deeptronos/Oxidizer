import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", help="Verbose output (Debug)", action="store_true")
parser.add_argument("file", help="The audio file (WAV) to be processed.", type=str)
parser.parse_args()
