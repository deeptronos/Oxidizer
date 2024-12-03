import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", help="The audio file (WAV) to be processed.", type=str)
parser.parse_args()
