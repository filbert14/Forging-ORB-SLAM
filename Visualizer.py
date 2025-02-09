import cv2
import os

import sys
import sdl2.ext

from time import sleep
import argparse

def play_sequence(sequence_path, window_size):
    sdl2.ext.init()

    window = sdl2.ext.Window(sequence_path, size=window_size)
    window.show()
    running = True

    factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)

    image_files = [image_file for image_file in sorted(os.listdir(sequence_path)) if not image_file == "timestamps.txt"]

    while running:
        for image_file in image_files:
            sprite         = factory.from_image(os.path.join(sequence_path, image_file))
            spriterenderer = factory.create_sprite_render_system(window)
            spriterenderer.render(sprite)

            # Handle events
            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    running = False
                    break

            if not running:
                break

            window.refresh()
            sleep(0.02)

    sdl2.ext.quit()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_path", required=True, help="Path to a directory containing images")
    ap.add_argument("--window_size", required=True, nargs=2, help="Size of the window, given in (W, H)")

    args = vars(ap.parse_args())
    sequence_path = args["sequence_path"]
    window_size   = args["window_size"]
    window_size   = (int(window_size[0]), int(window_size[1]))

    play_sequence(sequence_path, window_size)

if __name__ == "__main__":
    main()