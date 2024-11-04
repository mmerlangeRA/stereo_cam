from dataclasses import dataclass
import cv2

@dataclass
class AttentionWindow:
    left:int
    right:int
    top:int
    bottom:int
    makeMultipleOf8:bool=True

    def __init__(self,left:int, right:int, top:int, bottom:int, makeMultipleOf8=True) -> None:
        assert left < right and top < bottom, "Invalid window coordinates"
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        if makeMultipleOf8:
            self.makeItMultipleOf8()

    @property
    def width(self) -> int:
        return self.right - self.left
    
    @property
    def height(self) -> int:
        return self.bottom - self.top
    
    @property
    def center_x(self) -> int:
        return self.left + self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.top + self.height // 2
    
    @property
    def center(self) -> tuple[int, int]:
        return (self.center_x, self.center_y)
    
    @property
    def top_left(self) -> tuple[int, int]:
        return (self.left, self.top)
    @property
    def bottom_left(self) -> tuple[int, int]:
        return (self.left, self.bottom)
    
    @property
    def top_right(self) -> tuple[int, int]:
        return (self.right, self.top) 
    
    @property
    def bottom_right(self) -> tuple[int, int]:
        return (self.right, self.bottom)
    @property
    def top_center(self) -> tuple[int, int]:
        return (self.center_x, self.top)
    
    @property
    def bottom_center(self) -> tuple[int, int]:
        return (self.center_x, self.bottom)

    @property
    def area(self) -> int:
        return self.width * self.height

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
    
