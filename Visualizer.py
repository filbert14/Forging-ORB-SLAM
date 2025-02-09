import os
import tempfile
from time import sleep

import cv2
import sdl2.ext
from tqdm import tqdm

class Visualizer:
    def __init__(self):
        pass

    def save2D(self, images, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for i, image in enumerate(tqdm(images, desc="Saving images")):
            cv2.imwrite(os.path.join(save_dir, f"{i:06}.png"), image)

    def display2D(self, images, title="display2D", delay=0.02):
        sdl2.ext.init()

        size = images[0].shape
        size = (size[1], size[0])
        window = sdl2.ext.Window(title, size=size)

        factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Creating temporary directory: {temp_dir}")
            self.save2D(images, temp_dir)

            image_files = [image_file for
                           image_file in
                           sorted(os.listdir(temp_dir))]

            window.show()

            running = True
            while running:
                for image_file in image_files:
                    sprite = factory.from_image(os.path.join(temp_dir, image_file))
                    spriterenderer = factory.create_sprite_render_system(window)
                    spriterenderer.render(sprite)

                    events = sdl2.ext.get_events()
                    for event in events:
                        if event.type == sdl2.SDL_QUIT:
                            running = False
                            break

                    if not running:
                        break

                    window.refresh()
                    sleep(delay)

        sdl2.ext.quit()