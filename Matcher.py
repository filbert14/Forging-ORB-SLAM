import cv2

# @TODO:
# - [ ] Follow the actual matching criteria presented in the paper

class Matcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def SearchForInitialization(self, initial_frame, current_frame):
        return self.matcher.match(initial_frame.descriptors, current_frame.descriptors)
