
import numpy as np
import math
import pickle

import time
import os
import cv2

import ctypes

from src.gopro.libTransformer import Face, Plan, LookupTable, ImageTransformer, FrameSpecs

#Partie conversion en image sphérique
# Code inspiré de https://github.com/trek-view/max2sphere
# Explication de la projection GOPRO https://paulbourke.net/panorama/gopromax2sphere/

class Face:
    Invalid, Left, Right, Up, Front, Back, Down = range(-1, 6, 1)

class Plan:
    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

class FrameSpecs:
    def __init__(self, width=0, height=0, sidewidth=0, centerwidth=0, blendwidth=0, equiwidth=0, equiheight=0, antialias=0, name=''):
        self.width = width
        self.height = height
        self.sidewidth = sidewidth
        self.centerwidth = centerwidth
        self.blendwidth = blendwidth
        self.equiwidth = equiwidth
        self.equiheight = equiheight
        self.antialias = antialias
        self.name = name

class Transformer:
    def __init__(self):
        self.framespecs = FrameSpecs()
        self.sphericalImage = None
        self.LUT = None
        self.LUTsize = 0

    def load(self, specs):
        start_time = time.time()

        print(f'[ImageTransformer] load ...')
        self.framespecs = specs
        self.LUTsize = self.framespecs.equiwidth * self.framespecs.equiheight * self.framespecs.antialias * self.framespecs.antialias
        # [face, u , v, i1, j1, i2, j2, alpha]
        self.LUT = np.zeros([self.LUTsize, 8], dtype = np.double)

        if os.path.isfile('LUT_' + self.framespecs.name + '.dat'):
            start_time = time.time()
            with open('LUT_' + self.framespecs.name + '.dat', 'rb') as lutFile:
                self.LUT = pickle.load(lutFile)
                lutFile.close()                
            print("--- READ %s seconds ---" % (time.time() - start_time))
        else:
            plans = [Plan() for _ in range(6)]
            plans[Face.Left].a = -1
            plans[Face.Left].b = 0
            plans[Face.Left].c = 0
            plans[Face.Left].d = -1

            plans[Face.Right].a = 1
            plans[Face.Right].b = 0
            plans[Face.Right].c = 0
            plans[Face.Right].d = -1

            plans[Face.Down].a = 0
            plans[Face.Down].b = 0
            plans[Face.Down].c = 1
            plans[Face.Down].d = -1

            plans[Face.Up].a = 0
            plans[Face.Up].b = 0
            plans[Face.Up].c = -1
            plans[Face.Up].d = -1

            plans[Face.Front].a = 0
            plans[Face.Front].b = 1
            plans[Face.Front].c = 0
            plans[Face.Front].d = -1

            plans[Face.Back].a = 0
            plans[Face.Back].b = -1
            plans[Face.Back].c = 0
            plans[Face.Back].d = -1

            dx = self.framespecs.antialias * self.framespecs.equiwidth
            dy = self.framespecs.antialias * self.framespecs.equiheight
            itable = 0

            for j in range(self.framespecs.equiheight):
                y0 = j / self.framespecs.equiheight            
                for i in range(self.framespecs.equiwidth):
                    x0 = i / self.framespecs.equiwidth
                    for aj in range(self.framespecs.antialias):
                        y = y0 + aj / dy
                        for ai in range(self.framespecs.antialias):
                            x = x0 + ai / dx
                            longitude = x * (2 * math.pi) - math.pi
                            latitude = y * math.pi - (math.pi / 2)
                            if self.findFaceUV(longitude, latitude, itable, plans) == False:
                                print(f'[ImageTransformer] WARNING : findFaceUV error')
                                return
                            self.generatePixelMatrix(itable)
                            itable += 1

            print("--- INIT %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

            with open('LUT_' + self.framespecs.name + '.dat', 'wb') as lutFile:
                pickle.dump(self.LUT, lutFile, pickle.HIGHEST_PROTOCOL)
                lutFile.close()

            print("--- WRITE %s seconds ---" % (time.time() - start_time))

    def projectCubeMapToEAC(self, imageH, imageV, roi = None, drawDown = False, drawUp = False):        
        antialias_root = self.framespecs.antialias * self.framespecs.antialias
        itable = 0

        _roi = (0, 0, self.framespecs.equiwidth, self.framespecs.equiheight)
        
        if roi is not None:
            _roi = roi

        equi_image = np.zeros((_roi[3] - _roi[1], _roi[2] - _roi[0], 3), dtype=np.uint8)

        for j in range(_roi[1], _roi[3]):
            itable = (j * self.framespecs.equiwidth + _roi[0]) * antialias_root
            for i in range(_roi[0], _roi[2]):
                equi_image[j - _roi[1], i - _roi[0]] = self.getPixel(itable, imageH, imageV, drawDown, drawUp)
                itable += antialias_root
                # Je ne vois pas vraiment la difference avec ou sans antialising ???
                """equi_color = [0, 0, 0]
                for a in range(antialias_root):
                    color = self.getPixel(itable, imageH, imageV)
                    equi_color[0] += color[0]
                    equi_color[1] += color[1]
                    equi_color[2] += color[2]
                    itable += 1
                equi_color[:] =  [x / antialias_root for x in equi_color]
                equi_image[j, i] = equi_color"""

        return equi_image

    def findFaceUV(self, longitude, latitude, index, plans):
        face = Face.Invalid
        fourdivpi = 4.0 / math.pi

        coslatitude = math.cos(latitude)
        p = np.array([coslatitude * math.sin(longitude), coslatitude * math.cos(longitude), math.sin(latitude)])

        for k in range(6):
            denom = -(plans[k].a * p[0] + plans[k].b * p[1] + plans[k].c * p[2])

            if abs(denom) < 1e-6:
                continue

            mu = plans[k].d / denom
            if mu < 0:
                continue

            q = mu * p

            if k in [Face.Left, Face.Right] and -1 <= q[1] <= 1 and -1 <= q[2] <= 1:
                face = k
                q[1] = math.atan(q[1]) * fourdivpi
                q[2] = math.atan(q[2]) * fourdivpi
                break
            elif k in [Face.Front, Face.Back] and -1 <= q[0] <= 1 and -1 <= q[2] <= 1:
                face = k
                q[0] = math.atan(q[0]) * fourdivpi
                q[2] = math.atan(q[2]) * fourdivpi
                break
            elif k in [Face.Up, Face.Down] and -1 <= q[0] <= 1 and -1 <= q[1] <= 1:
                face = k
                q[0] = math.atan(q[0]) * fourdivpi
                q[1] = math.atan(q[1]) * fourdivpi
                break

        if face == Face.Invalid:
            print("[ERROR] findFaceUV() - Didn't find an intersecting face, shouldn't happen!")
            return False

        u = 0
        v = 0

        if face == Face.Left:
            u = q[1] + 1
            v = q[2] + 1
        elif face == Face.Right:
            u = 1 - q[1]
            v = q[2] + 1
        elif face == Face.Front:
            u = q[0] + 1
            v = q[2] + 1
        elif face == Face.Back:
            u = 1 - q[0]
            v = q[2] + 1
        elif face == Face.Up:
            u = 1 - q[0]
            v = 1 - q[1]
        elif face == Face.Down:
            u = 1 - q[0]
            v = q[1] + 1

        u *= 0.5
        v *= 0.5

        if u >= 1:
            u = 0.9999
        if v >= 1:
            v = 0.9999

        if not (0 <= u < 1 and 0 <= v < 1):
            print(f"[ERROR] findFaceUV() - Illegal (u,v) coordinate ({u}, {v}) on face {face}")
            return False
        
        self.LUT[index, 0] = face
        self.LUT[index, 1] = u
        self.LUT[index, 2] = v

        return True
    
    def generatePixelMatrix(self, index):
        face = self.LUT[index, 0]
        u = self.LUT[index, 1]
        v = self.LUT[index, 2]
        
        # Rotate u,v counterclockwise by 90 degrees for lower frame
        if face in [Face.Down, Face.Back, Face.Up]:
            u = self.LUT[index, 2]
            v = 0.9999 - self.LUT[index, 1]

        u_left = u
        u_right = u
        
        if face == Face.Front:
            x0 = self.framespecs.sidewidth
            w = self.framespecs.centerwidth
            self.LUT[index, 3] = x0 + u * w
            self.LUT[index, 4] = v * self.framespecs.height
            self.LUT[index, 7] = -1

        elif face == Face.Back:
            x0 = self.framespecs.sidewidth
            w = self.framespecs.centerwidth
            self.LUT[index, 3] = max(0, self.framespecs.width - 1 - int(x0 + u * w))
            self.LUT[index, 4] = max(0, self.framespecs.height - 1 - int(v * self.framespecs.height))
            self.LUT[index, 7] = -1

        elif face == Face.Left:
            w = self.framespecs.sidewidth
            duv = self.framespecs.blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                self.LUT[index, 3] = u_left * w
                self.LUT[index, 4] = v * self.framespecs.height
                self.LUT[index, 7] = -1
            elif u_right >= 0.5 + 2 * duv:
                self.LUT[index, 3] = u_right * w
                self.LUT[index, 4] = v * self.framespecs.height
                self.LUT[index, 7] = -1
            else:
                self.LUT[index, 3] = u_left * w
                self.LUT[index, 4] = v * self.framespecs.height
                self.LUT[index, 5] = u_right * w
                self.LUT[index, 6] = v * self.framespecs.height
                self.LUT[index, 7] = (u_left - 0.5 + 2 * duv) / (2 * duv)

        elif face == Face.Up:
            w = self.framespecs.sidewidth
            duv = self.framespecs.blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                self.LUT[index, 3] = max(0, self.framespecs.width - 1 - int(u_left * w))
                self.LUT[index, 4] = max(0, self.framespecs.height- 1 - int(v * self.framespecs.height))
                self.LUT[index, 7] = -1
            elif u_right >= 0.5 + 2 * duv:
                self.LUT[index, 3] = max(0, self.framespecs.width - 1 - int(u_right * w))
                self.LUT[index, 4] = max(0, self.framespecs.height - 1 - int(v * self.framespecs.height))
                self.LUT[index, 7] = -1
            else:
                self.LUT[index, 3] = max(0, self.framespecs.width - 1 - int(u_left * w))
                self.LUT[index, 4] = max(0, self.framespecs.height - 1 - int(v * self.framespecs.height))
                self.LUT[index, 5] = max(0, self.framespecs.width - 1 - int(u_right * w))
                self.LUT[index, 6] = max(0, self.framespecs.height - 1 - int(v * self.framespecs.height))
                self.LUT[index, 7] = (u_left - 0.5 + 2 * duv) / (2 * duv)

        elif face == Face.Right:
            x0 = self.framespecs.sidewidth + self.framespecs.centerwidth
            w = self.framespecs.sidewidth
            duv = self.framespecs.blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv
            if u_left <= 0.5 - 2 * duv:
                self.LUT[index, 3] = int(x0 + u_left * w)
                self.LUT[index, 4] = int(v * self.framespecs.height)
                self.LUT[index, 7] = -1
            elif u_right >= 0.5 + 2 * duv:
                self.LUT[index, 3] = int(x0 + u_right * w)
                self.LUT[index, 4] = int(v * self.framespecs.height)
                self.LUT[index, 7] = -1
            else:
                self.LUT[index, 3] = int(x0 + u_left * w)
                self.LUT[index, 4] = int(v * self.framespecs.height)
                self.LUT[index, 5] = int(x0 + u_right * w)
                self.LUT[index, 6] = int(v * self.framespecs.height)
                self.LUT[index, 7] = (u_left - 0.5 + 2 * duv) / (2 * duv)

        elif face == Face.Down:
            x0 = self.framespecs.sidewidth + self.framespecs.centerwidth
            w = self.framespecs.sidewidth
            duv = self.framespecs.blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv
            if u_left <= 0.5 - 2 * duv:
                self.LUT[index, 3] = max(0, self.framespecs.width - 1 - int(x0 + u_left * w))
                self.LUT[index, 4] = max(0, self.framespecs.height - 1 - int(v * self.framespecs.height))
                self.LUT[index, 7] = -1
            elif u_right >= 0.5 + 2 * duv:
                self.LUT[index, 3] = max(0, self.framespecs.width - 1 - int(x0 + u_right * w))
                self.LUT[index, 4] = max(0, self.framespecs.height - 1 - int(v * self.framespecs.height))
                self.LUT[index, 7] = -1
            else:
                self.LUT[index, 3] = max(0, self.framespecs.width - 1 - int(x0 + u_left * w))
                self.LUT[index, 4] = max(0, self.framespecs.height - 1 - int(v * self.framespecs.height))
                self.LUT[index, 5] = max(0, self.framespecs.width - 1 - int(x0 + u_right * w))
                self.LUT[index, 6] = max(0, self.framespecs.height - 1 - int(v * self.framespecs.height))
                self.LUT[index, 7] = (u_left - 0.5 + 2 * duv) / (2 * duv)
        else:
            print(f'[ImageTransformer] WARNING : unknown face {face}')
                
    def getPixel(self, index, imageH, imageV, drawDown = False, drawUp = False):        
        c = [0, 0, 0]

        face = self.LUT[index, 0]

        if face == Face.Front:
            c = imageH[int(self.LUT[index, 4]), int(self.LUT[index, 3])]

        elif face == Face.Left:
            if self.LUT[index, 7] < 0:
                c = imageH[int(self.LUT[index, 4]), int(self.LUT[index, 3])]
            else:
                c1 = imageH[int(self.LUT[index, 4]), int(self.LUT[index, 3])]
                c2 = imageH[int(self.LUT[index, 6]), int(self.LUT[index, 5])]
                c = self.colorBlend(c1, c2, self.LUT[index, 7]) 

        elif face == Face.Right:
            if self.LUT[index, 7] < 0:
                c = imageH[int(self.LUT[index, 4]), int(self.LUT[index, 3])]
            else:
                c1 = imageH[int(self.LUT[index, 4]), int(self.LUT[index, 3])]
                c2 = imageH[int(self.LUT[index, 6]), int(self.LUT[index, 5])]
                c = self.colorBlend(c1, c2, self.LUT[index, 7]) 

        elif face == Face.Back:
            c = imageV[int(self.LUT[index, 4]), int(self.LUT[index, 3])]

        if drawDown:
            if face == Face.Up:
                if self.LUT[index, 7] < 0:
                    c = imageV[int(self.LUT[index, 4]), int(self.LUT[index, 3])]
                else:
                    c1 = imageV[int(self.LUT[index, 4]), int(self.LUT[index, 3])]
                    c2 = imageV[int(self.LUT[index, 6]), int(self.LUT[index, 5])]
                    c = self.colorBlend(c1, c2, self.LUT[index, 7])

        if drawUp:
            if face == Face.Down:
                if self.LUT[index, 7] < 0:
                    c = imageV[int(self.LUT[index, 4]), int(self.LUT[index, 3])]
                else:
                    c1 = imageV[int(self.LUT[index, 4]), int(self.LUT[index, 3])]
                    c2 = imageV[int(self.LUT[index, 6]), int(self.LUT[index, 5])]
                    c = self.colorBlend(c1, c2, self.LUT[index, 7])

        return c

    def colorBlend(self, c1, c2, alpha):
        alpha_inv = 1 - alpha
        r = alpha_inv * c1[0] + alpha * c2[0]
        g = alpha_inv * c1[1] + alpha * c2[1]
        b = alpha_inv * c1[2] + alpha * c2[2]
        return (int(r), int(g), int(b))
    
    def translateToCenter(self, offsets, image):
        if len(offsets) != 2:
            print('[ImageTransformer] ERROR : offset size is not 2')
            return image
        
        center = image.shape[1] / 2
        dx = int(offsets[0] - center)
        part_image_l = image[:, :dx]
        part_image_r = image[:, dx:]
        dx = part_image_r.shape[1]
        image_out = np.zeros([image.shape[0], image.shape[1], 3],dtype=np.uint8)
        image_out[:, :dx] = part_image_r
        image_out[:, dx:] = part_image_l
        return image_out
    
    def loadFrameSpecsFromGoProMax(self):
        # GOPRO Max 5K specs
        fs = FrameSpecs()
        fs.width = 4096
        fs.height = 1344
        fs.blendwidth = 32
        fs.sidewidth = fs.height + fs.blendwidth
        fs.centerwidth = fs.height
        fs.equiwidth = int(fs.width + fs.height - (fs.blendwidth * 2))
        fs.equiheight = int(fs.equiwidth / 2)
        fs.antialias = 2
        fs.name = 'gopro_max_5k'
        return fs
    
class Transformer_optimized:
    def __init__(self) -> None:
        self.transform = None
        self.framespecs = FrameSpecs()
        TransformerHandle = ctypes.POINTER(ctypes.c_char)
        #self.lib = ctypes.CDLL("./ImageTransformerLib.dll")
        #lib.add.argtypes = (ctypes.c_int, ctypes.c_int)
        #lib.add.restype = ctypes.c_int
        self.lib = ImageTransformer()
        #self.lib.createInstance.argtypes = []
        '''
        self.lib.createInstance.restype = TransformerHandle
        self.lib.releaseInstance.argtypes = [TransformerHandle]
        self.lib.loadGoProMax.argtypes = [TransformerHandle]
        self.lib.loadGoProMax.restype = ctypes.c_bool
        self.lib.framespecs_width.argtypes = [TransformerHandle]
        self.lib.framespecs_width.restype = ctypes.c_int
        self.lib.framespecs_height.argtypes = [TransformerHandle]
        self.lib.framespecs_height.restype = ctypes.c_int
        self.lib.framespecs_centerwidth.argtypes = [TransformerHandle]
        self.lib.framespecs_centerwidth.restype = ctypes.c_int
        self.lib.framespecs_blendwidth.argtypes = [TransformerHandle]
        self.lib.framespecs_blendwidth.restype = ctypes.c_int
        self.lib.framespecs_equiwidth.argtypes = [TransformerHandle]
        self.lib.framespecs_equiwidth.restype = ctypes.c_int
        self.lib.framespecs_equiheight.argtypes = [TransformerHandle]
        self.lib.framespecs_equiheight.restype = ctypes.c_int
        self.lib.framespecs_antialias.argtypes = [TransformerHandle]
        self.lib.framespecs_antialias.restype = ctypes.c_int
        self.lib.framespecs_sidewidth.argtypes = [TransformerHandle]
        self.lib.framespecs_sidewidth.restype = ctypes.c_int
        np_pointer = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags="C")
        self.lib.cubemapToSpheric.argtypes = [TransformerHandle, 
                                              np_pointer, ctypes.c_int, ctypes.c_int, 
                                              np_pointer, ctypes.c_int, ctypes.c_int, 
                                              np_pointer, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_bool, ctypes.c_bool]
        '''
    def loadFrameSpecsFromGoProMax(self):
        # GOPRO Max 5K specs
        fs = FrameSpecs()
        fs.width = 4096
        fs.height = 1344
        fs.blendwidth = 32
        fs.sidewidth = fs.height + fs.blendwidth
        fs.centerwidth = fs.height
        fs.equiwidth = int(fs.width + fs.height - (fs.blendwidth * 2))
        fs.equiheight = int(fs.equiwidth / 2)
        fs.antialias = 2
        fs.name = 'gopro_max_5k'
        return fs
    
    def load(self):
        self.transform = self.lib.createInstance()
        self.lib.loadGoProMax(self.transform)
        self.framespecs.width = self.lib.framespecs_width(self.transform)
        self.framespecs.height = self.lib.framespecs_height(self.transform)
        self.framespecs.centerwidth = self.lib.framespecs_centerwidth(self.transform)
        self.framespecs.blendwidth = self.lib.framespecs_blendwidth(self.transform)
        self.framespecs.equiwidth = self.lib.framespecs_equiwidth(self.transform)
        self.framespecs.equiheight = self.lib.framespecs_equiheight(self.transform)
        self.framespecs.antialias = self.lib.framespecs_antialias(self.transform)
        self.framespecs.sidewidth = self.lib.framespecs_sidewidth(self.transform)

    def release(self):
        self.lib.releaseInstance(self.transform)
        self.transform = None

    def projectCubeMapToEAC(self, imageH, imageV, roi = None, drawDown = False, drawUp = False):
        _roi = (0, 0, self.framespecs.equiwidth, self.framespecs.equiheight)
        if roi is not None:
            _roi = roi
        imageS = np.zeros((_roi[3] - _roi[1], _roi[2] - _roi[0], 3), dtype=np.uint8)
        self.lib.cubemapToSpheric(
                                  imageH,
                                  imageV,  
                                  imageS, 
                                  _roi,
                                  drawDown,
                                  drawUp)
        return imageS
        
    def translateToCenter(self, offsets, image):
        if len(offsets) != 2:
            print('[ImageTransformer] ERROR : offset size is not 2')
            return image
        
        center = image.shape[1] / 2
        dx = int(offsets[0] - center)
        part_image_l = image[:, :dx]
        part_image_r = image[:, dx:]
        dx = part_image_r.shape[1]
        image_out = np.zeros([image.shape[0], image.shape[1], 3],dtype=np.uint8)
        image_out[:, :dx] = part_image_r
        image_out[:, dx:] = part_image_l
        return image_out
    

if __name__ == '__main__':

    transformer = Transformer_optimized()
    specs:FrameSpecs = transformer.loadFrameSpecsFromGoProMax()
    specs.equiheight=2388
    specs.equiwidth=5376
    
    specs.equiheight=597
    specs.equiwidth=1344

    transformer.lib.load(specs)

        # Assume imageH and imageV are your horizontal and vertical cubemap images loaded using cv2
    imageH = cv2.imread(r"C:\Users\mmerl\projects\stereo_cam\Photos\P1\D_P1_CAM_G_0_CUBE.png")
    imageV = cv2.imread(r"C:\Users\mmerl\projects\stereo_cam\Photos\P1\D_P1_CAM_D_0_CUBE.png")
    imageS = np.zeros((specs.equiheight, specs.equiwidth, 3), dtype=np.uint8)
    roi = None  # Define ROI if needed
    drawDown = True
    drawTop = True
    spherical_image = transformer.projectCubeMapToEAC(imageH, imageV, roi, drawDown, drawTop)
    cv2.imshow("Spherical Image", spherical_image)
