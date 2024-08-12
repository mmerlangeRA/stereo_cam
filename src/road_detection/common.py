import cv2


class AttentionWindow:
    left:int
    right:int
    top:int
    bottom:int
    def __init__(self,left:int, right:int, top:int, bottom:int) -> None:
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.makeItMultipleOf8()

    def makeItMultipleOf8(self) -> None:
        '''
        Ensure width and height are multiple of 8
        '''
        # Adjust width (right - left)
        width = self.right - self.left
        if width % 8 != 0:
            # Increase right to make width a multiple of 4
            adjustment = 8 - (width % 8)
            self.right += adjustment

        # Adjust height (bottom - top)
        height = self.bottom - self.top
        if height % 8 != 0:
            # Increase bottom to make height a multiple of 4
            adjustment = 8 - (height % 8)
            self.bottom += adjustment
    
    def crop_image(self, img:cv2.typing.MatLike) -> cv2.typing.MatLike:
        return img[self.top:self.bottom, self.left:self.right]
    
    def __str__(self) -> str:
        return f"AttentionWindow(left={self.left}, right={self.right}, top={self.top}, bottom={self.bottom})"
