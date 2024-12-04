import argparse
import scipy.io.wavfile as wav
import numpy as np


def add_dc_offset(input_file, output_file, offset, verbose: bool):
    if verbose:
        print(f"add_dc_offset({input_file}, {output_file}, {offset}, {verbose})")
    rate, data = wav.read(input_file)
    # Determine data type and limits
    if data.dtype == np.int16:
        info = np.iinfo(np.int16)
    elif data.dtype == np.int32:
        info = np.iinfo(np.int32)
    elif data.dtype == np.float32:
        info = np.finfo(np.float32)
    else:
        raise ValueError("Unsupported data type: {}".format(data.dtype))

    if verbose:
        print(f"Sample Rate: {rate}")
        print(f"Data Shape: {data.shape}")

    data = data + offset  # Add DC offset

    data = np.clip(data, info.min, info.max)  # Ensure data is within valid range

    wav.write(output_file, rate, data.astype(data.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", help="Verbose output mode.", action="store_true")
    parser.add_argument(
        "infile", help="The audio file (WAV) to be processed.", type=str
    )
    parser.add_argument(
        "outfile", help="The name of the audio file (WAV) to be written.", type=str
    )
    parser.add_argument("dc_offset", help="The DC offset to apply.", type=int)
    args = parser.parse_args()
    if args.v:
        print(f"args: {args}")

    add_dc_offset(args.infile, args.outfile, args.dc_offset, args.v)


if __name__ == "__main__":  # Entrypoint
    main()
