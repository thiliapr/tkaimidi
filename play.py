import mido
import math
import pathlib
import argparse
from time import sleep
from utils import midi_to_notes, norm_data
from model import TIME_PRECISION
from generate import model_output_to_track


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("-s", "--strict", action="store_true", default=False, help="规范化数据时启用strict模式。因为需要尽量保留原来的音符时间差距，所以音符时间差距可能很大，导致有许多填充音符。")
    parser.add_argument("-m", "--medium-mote", type=int, default=72, help="将音高的平均值调整到指定数值。范围: 0-127")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    import pygame.midi

    pygame.init()
    pygame.midi.init()
    player = pygame.midi.Output(0)
    playlist: list[pathlib.Path] = list(args.path.glob("**/*.mid")) if args.path.is_dir() else [args.path]
    while True:
        for mid in playlist:
            print(mid.relative_to(args.path) if args.path.is_dir() else mid)
            notes = [(note, time) for _, note, time, _ in midi_to_notes(mido.MidiFile(mid, clip=True))]
            notes = norm_data(notes, TIME_PRECISION, 960, strict=args.strict)
            notes_offest = max(0, int(args.medium_mote - sum(note for note, _ in notes) / len(notes)))
            for i, (note, time) in enumerate(notes):
                note = note + notes_offest
                if note > 127:
                    note -= math.ceil((note - 127) / 12) * 12
                notes[i] = (note, time)

            track = model_output_to_track(notes)
            file = mido.MidiFile(tracks=[track])
            for message in file.play():
                if message.is_meta:
                    continue
                player.write_short(message.bytes()[0], message.note, message.velocity)
            sleep(4)


if __name__ == "__main__":
    main()
