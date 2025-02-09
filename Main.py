import yaml

from EDSLoader import EDSLoader
from Tracking  import Tracking

def main():
    with open("Settings.yaml", "r") as file:
        settings = yaml.safe_load(file)

    edsloader = EDSLoader("EDS", "09_ziggy_flying_pieces")
    tracking  = Tracking(edsloader.images, settings)
    tracking.GrabImage()

if __name__ == "__main__":
    main()
