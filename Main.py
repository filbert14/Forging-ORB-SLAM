from EDSLoader import EDSLoader
from Tracking  import Tracking

def main():
    edsloader = EDSLoader("EDS", "09_ziggy_flying_pieces")
    tracking  = Tracking(edsloader.images)

if __name__ == "__main__":
    main()
