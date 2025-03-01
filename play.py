import pathlib
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("-s", "--strict", action="store_true", default=False, help="规范化数据时启用strict模式。因为需要尽量保留原来的音符时间差距，所以音符时间差距可能很大，导致有许多填充音符。")
    parser.add_argument("-m", "--medium-mote", type=int, default=72, help="将音高的平均值调整到指定数值。范围: 0-127")
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    pygame.midi.init()
    player = pygame.midi.Output(0)
    playlist: list[pathlib.Path] = list(args.path.glob("**/*.mid")) if args.path.is_dir() else [args.path]
    while True:
        for mid in playlist:
            print(mid.relative_to(args.path) if args.path.is_dir() else mid)
            notes = [(note, time) for _, note, time, _ in midi_to_notes(mido.MidiFile(mid, clip=True))]
            notes = normalize_times(notes, TIME_PRECISION, strict=args.strict)
            notes_offest = max(0, int(args.medium_mote - sum(note for note, _ in notes) / len(notes)))
            for i, (note, time) in enumerate(notes):
                note = note + notes_offest
                if note > MAX_NOTE:
                    note -= math.ceil((note + 1 - MAX_NOTE) / 12) * 12
                notes[i] = (note, time)

            track = model_output_to_track(notes_to_note_intervals(notes, MAX_NOTE + 1))
            file = mido.MidiFile(tracks=[track])
            for message in file.play():
                if not message.type.startswith("note_o"):
                    continue
                player.write_short(message.bytes()[0], message.note, message.velocity)
            sleep(4)


if __name__ == "__main__":
    args = parse_args()

    import mido
    import math
    import pygame.midi
    from time import sleep
    from utils import midi_to_notes, normalize_times, notes_to_note_intervals
    from model import TIME_PRECISION, MAX_NOTE
    from generate import model_output_to_track

    main(args)
