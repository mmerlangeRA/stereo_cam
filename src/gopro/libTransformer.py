import numpy as np
import cv2
import math
from enum import Enum
from numba import cuda, njit, prange

# Define constants
M_PI = math.pi
M_PI_2 = math.pi / 2

class Face(Enum):
    Left = 0
    Right = 1
    Top = 2
    Down = 3
    Front = 4
    Back = 5
    Invalid = 6

class Plan:
    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

class LookupTable:
    def __init__(self):
        self.u = 0.0
        self.v = 0.0
        self.face = Face.Invalid.value
        self.ix = 0
        self.iy = 0
        self.jx = 0
        self.jy = 0
        self.alpha = -1.0

class FrameSpecs:
    def __init__(self, antialias=0, blendwidth=0, centerwidth=0, equiwidth=0, equiheight=0, width=0, height=0, name="", sidewidth=0):
        self.antialias = antialias
        self.blendwidth = blendwidth
        self.centerwidth = centerwidth
        self.equiwidth = equiwidth
        self.equiheight = equiheight
        self.width = width
        self.height = height
        self.name = name
        self.sidewidth = sidewidth

class ImageTransformer:
    m_framespecs:FrameSpecs 
    def __init__(self):
        self.m_LUT = None
        self.m_LUTsize = 0
        self.m_framespecs = None

    def frameSpecs(self):
        return self.m_framespecs

    def loadFrameSpecsFromGoProMax(self):
        fs = FrameSpecs()
        fs.width = 4096
        fs.height = 1344
        fs.blendwidth = 32
        fs.sidewidth = fs.height + fs.blendwidth
        fs.centerwidth = fs.height
        fs.equiwidth = int(fs.width + fs.height - (fs.blendwidth * 2))
        fs.equiheight = int(fs.equiwidth / 2)
        fs.antialias = 2
        fs.name = "gopro_max_5k"
        return fs

    def load(self, specs:FrameSpecs):
        self.m_framespecs = specs
        self.m_LUTsize = specs.equiwidth * specs.equiheight * specs.antialias * specs.antialias

        # Initialize plans for the 6 cube faces
        plans = [Plan() for _ in range(6)]
        plans[Face.Left.value].a = -1; plans[Face.Left.value].b = 0; plans[Face.Left.value].c = 0; plans[Face.Left.value].d = -1
        plans[Face.Right.value].a = 1; plans[Face.Right.value].b = 0; plans[Face.Right.value].c = 0; plans[Face.Right.value].d = -1
        plans[Face.Top.value].a = 0; plans[Face.Top.value].b = 0; plans[Face.Top.value].c = 1; plans[Face.Top.value].d = -1
        plans[Face.Down.value].a = 0; plans[Face.Down.value].b = 0; plans[Face.Down.value].c = -1; plans[Face.Down.value].d = -1
        plans[Face.Front.value].a = 0; plans[Face.Front.value].b = 1; plans[Face.Front.value].c = 0; plans[Face.Front.value].d = -1
        plans[Face.Back.value].a = 0; plans[Face.Back.value].b = -1; plans[Face.Back.value].c = 0; plans[Face.Back.value].d = -1

        # Initialize LUT
        self.m_LUT = [LookupTable() for _ in range(self.m_LUTsize)]
        dx = specs.antialias * specs.equiwidth
        dy = specs.antialias * specs.equiheight

        # Generate the LUT
        self.generate_LUT(plans, dx, dy,specs.equiwidth,specs.equiheight, specs.antialias, specs.sidewidth,specs.centerwidth,specs.width,specs.height,specs.blendwidth)

    def generate_LUT(self, plans, dx, dy,equiwidth,equiheight,antialias,sidewidth,centerwidth,width,height,blendwidth):
        # Check if CUDA is available
        if cuda.is_available():
            print("[ImageTransformer] CUDA is available. Using GPU acceleration.")
            self.generate_LUT_cuda(plans, dx, dy,equiwidth,equiheight,antialias,sidewidth,centerwidth,width,height,blendwidth)
        else:
            print("[ImageTransformer] CUDA is not available. Using CPU.")
            self.generate_LUT_cpu(plans, dx, dy,sidewidth,centerwidth,width,height,blendwidth)

    def generate_LUT_cpu(self, specs, plans, dx, dy):
        index = 0
        for j in range(specs.equiheight):
            y0 = j / specs.equiheight
            for i in range(specs.equiwidth):
                x0 = i / specs.equiwidth
                for aj in range(specs.antialias):
                    y = y0 + aj / dy
                    for ai in range(specs.antialias):
                        x = x0 + ai / dx
                        longitude = x * (2 * M_PI) - M_PI
                        latitude = y * M_PI - M_PI_2
                        success = self.findFaceUV(longitude, latitude, index, plans)
                        if not success:
                            print(f"[ImageTransformer] Error in findFaceUV at index {index}")
                            return False
                        self.generatePixelMatrix(index)
                        index += 1

    def generate_LUT_cuda(self, plans, dx, dy,equiwidth,equiheight,antialias,sidewidth,centerwidth,width,height,blendwidth):
        # Convert plans to numpy arrays for CUDA
        plans_array = np.array([[p.a, p.b, p.c, p.d] for p in plans], dtype=np.float64)
        m_LUT_array = np.zeros((self.m_LUTsize, 8), dtype=np.float64)  # u, v, face, ix, iy, jx, jy, alpha

        # Prepare grid and block dimensions
        threadsperblock = 256
        blockspergrid = (self.m_LUTsize + (threadsperblock - 1)) // threadsperblock

        # Call CUDA kernel
        self.cuda_generate_LUT[blockspergrid, threadsperblock](equiwidth, equiheight, antialias, dx, dy, plans_array, m_LUT_array, sidewidth,centerwidth,width,height,blendwidth)

        # Convert m_LUT_array back to list of LookupTable
        for idx in range(self.m_LUTsize):
            lut = LookupTable()
            lut.u = m_LUT_array[idx, 0]
            lut.v = m_LUT_array[idx, 1]
            lut.face = int(m_LUT_array[idx, 2])
            lut.ix = int(m_LUT_array[idx, 3])
            lut.iy = int(m_LUT_array[idx, 4])
            lut.jx = int(m_LUT_array[idx, 5])
            lut.jy = int(m_LUT_array[idx, 6])
            lut.alpha = m_LUT_array[idx, 7]
            self.m_LUT[idx] = lut

    @staticmethod
    @cuda.jit
    def cuda_generate_LUT(equiwidth, equiheight, antialias, dx, dy, plans, m_LUT_array,sidewidth,centerwidth,width,height,blendwidth):
        idx = cuda.grid(1)
        total = equiwidth * equiheight * antialias * antialias
        if idx < total:
            j = idx // (equiwidth * antialias * antialias)
            i = (idx % (equiwidth * antialias * antialias)) // (antialias * antialias)
            aj = (idx % (antialias * antialias)) // antialias
            ai = idx % antialias

            y0 = j / equiheight
            x0 = i / equiwidth
            y = y0 + aj / dy
            x = x0 + ai / dx
            longitude = x * (2 * math.pi) - math.pi
            latitude = y * math.pi - (math.pi / 2)

            # Call findFaceUV
            face, u, v = ImageTransformer.cuda_findFaceUV(longitude, latitude, plans)
            m_LUT_array[idx, 0] = u
            m_LUT_array[idx, 1] = v
            m_LUT_array[idx, 2] = face

            # Call generatePixelMatrix
            ix, iy, jx, jy, alpha = ImageTransformer.cuda_generatePixelMatrix(face, u, v, sidewidth,centerwidth,width,height,blendwidth)
            m_LUT_array[idx, 3] = ix
            m_LUT_array[idx, 4] = iy
            m_LUT_array[idx, 5] = jx
            m_LUT_array[idx, 6] = jy
            m_LUT_array[idx, 7] = alpha

    @staticmethod
    @njit
    def cuda_findFaceUV(longitude, latitude, plans):
        face = Face.Invalid.value
        fourdivpi = 4.0 / math.pi
        coslatitude = math.cos(latitude)
        p = np.array([coslatitude * math.sin(longitude),
                      coslatitude * math.cos(longitude),
                      math.sin(latitude)], dtype=np.float64)
        for k in range(6):
            denom = -(plans[k, 0] * p[0] + plans[k, 1] * p[1] + plans[k, 2] * p[2])
            if denom == 0:
                continue
            mu = plans[k, 3] / denom
            if mu < 0:
                continue
            q = mu * p
            if k == Face.Left.value or k == Face.Right.value:
                if q[1] <= 1 and q[1] >= -1 and q[2] <= 1 and q[2] >= -1:
                    face = k
                q[1] = math.atan(q[1]) * fourdivpi
                q[2] = math.atan(q[2]) * fourdivpi
            elif k == Face.Front.value or k == Face.Back.value:
                if q[0] <= 1 and q[0] >= -1 and q[2] <= 1 and q[2] >= -1:
                    face = k
                q[0] = math.atan(q[0]) * fourdivpi
                q[2] = math.atan(q[2]) * fourdivpi
            elif k == Face.Top.value or k == Face.Down.value:
                if q[0] <= 1 and q[0] >= -1 and q[1] <= 1 and q[1] >= -1:
                    face = k
                q[0] = math.atan(q[0]) * fourdivpi
                q[1] = math.atan(q[1]) * fourdivpi
            if face != Face.Invalid.value:
                break

        if face == Face.Invalid.value:
            # Error handling
            pass

        if face == Face.Left.value:
            u = q[1] + 1
            v = q[2] + 1
        elif face == Face.Right.value:
            u = 1 - q[1]
            v = q[2] + 1
        elif face == Face.Front.value:
            u = q[0] + 1
            v = q[2] + 1
        elif face == Face.Back.value:
            u = 1 - q[0]
            v = q[2] + 1
        elif face == Face.Down.value:
            u = 1 - q[0]
            v = 1 - q[1]
        elif face == Face.Top.value:
            u = 1 - q[0]
            v = q[1] + 1

        u *= 0.5
        v *= 0.5

        if u >= 1:
            u = 0.9999
        if v >= 1:
            v = 0.9999

        return face, u, v

    @staticmethod
    @njit
    def cuda_generatePixelMatrix(face, u, v,sidewidth,centerwidth,width,height,blendwidth ):
        face_enum = Face(face)
        if face_enum == Face.Down or face_enum == Face.Back or face_enum == Face.Top:
            u, v = v, 0.9999 - u

        u_left = u
        u_right = u

        x0 = 0
        w = 0
        duv = 0.0
        ix = iy = jx = jy = 0
        alpha = -1.0

        if face_enum == Face.Front:
            x0 = sidewidth
            w = centerwidth
            ix = int(x0 + u * w)
            iy = int(v * height)
            alpha = -1.0
        elif face_enum == Face.Back:
            x0 = sidewidth
            w = centerwidth
            ix = int(max(0, width - 1 - (x0 + u * w)))
            iy = int(max(0, height - 1 - (v * height)))
            alpha = -1.0
        elif face_enum == Face.Left:
            w = sidewidth
            duv = blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                ix = int(u_left * w)
                iy = int(v * height)
                alpha = -1.0
            elif u_right >= 0.5 + 2 * duv:
                ix = int(u_right * w)
                iy = int(v * height)
                alpha = -1.0
            else:
                ix = int(u_left * w)
                iy = int(v * height)
                jx = int(u_right * w)
                jy = iy
                alpha = (u_left - 0.5 + 2 * duv) / (2 * duv)
        elif face_enum == Face.Down:
            w = sidewidth
            duv = blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                ix = int(max(0, width - 1 - (u_left * w)))
                iy = int(max(0, height - 1 - (v * height)))
                alpha = -1.0
            elif u_right >= 0.5 + 2 * duv:
                ix = int(max(0, width - 1 - (u_right * w)))
                iy = int(max(0, height - 1 - (v * height)))
                alpha = -1.0
            else:
                ix = int(max(0, width - 1 - (u_left * w)))
                iy = int(max(0, height - 1 - (v * height)))
                jx = int(max(0, width - 1 - (u_right * w)))
                jy = iy
                alpha = (u_left - 0.5 + 2 * duv) / (2 * duv)
        elif face_enum == Face.Right:
            x0 = sidewidth + centerwidth
            w = sidewidth
            duv = blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                ix = int(x0 + u_left * w)
                iy = int(v * height)
                alpha = -1.0
            elif u_right >= 0.5 + 2 * duv:
                ix = int(x0 + u_right * w)
                iy = int(v * height)
                alpha = -1.0
            else:
                ix = int(x0 + u_left * w)
                iy = int(v * height)
                jx = int(x0 + u_right * w)
                jy = iy
                alpha = (u_left - 0.5 + 2 * duv) / (2 * duv)
        elif face_enum == Face.Top:
            x0 = sidewidth + centerwidth
            w = sidewidth
            duv = blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                ix = int(max(0, width - 1 - (x0 + u_left * w)))
                iy = int(max(0, height - 1 - (v * height)))
                alpha = -1.0
            elif u_right >= 0.5 + 2 * duv:
                ix = int(max(0, width - 1 - (x0 + u_right * w)))
                iy = int(max(0, height - 1 - (v * height)))
                alpha = -1.0
            else:
                ix = int(max(0, width - 1 - (x0 + u_left * w)))
                iy = int(max(0, height - 1 - (v * height)))
                jx = int(max(0, width - 1 - (x0 + u_right * w)))
                jy = iy
                alpha = (u_left - 0.5 + 2 * duv) / (2 * duv)
        else:
            # Unknown face
            pass

        return ix, iy, jx, jy, alpha

    def findFaceUV(self, longitude, latitude, index, plans):
        face = Face.Invalid
        fourdivpi = 4.0 / M_PI
        coslatitude = math.cos(latitude)
        p = np.array([coslatitude * math.sin(longitude),
                      coslatitude * math.cos(longitude),
                      math.sin(latitude)])
        for k in range(6):
            denom = -(plans[k].a * p[0] + plans[k].b * p[1] + plans[k].c * p[2])
            if denom == 0:
                continue
            mu = plans[k].d / denom
            if mu < 0:
                continue
            q = mu * p
            if k == Face.Left.value or k == Face.Right.value:
                if q[1] <= 1 and q[1] >= -1 and q[2] <= 1 and q[2] >= -1:
                    face = Face(k)
                q[1] = math.atan(q[1]) * fourdivpi
                q[2] = math.atan(q[2]) * fourdivpi
            elif k == Face.Front.value or k == Face.Back.value:
                if q[0] <= 1 and q[0] >= -1 and q[2] <= 1 and q[2] >= -1:
                    face = Face(k)
                q[0] = math.atan(q[0]) * fourdivpi
                q[2] = math.atan(q[2]) * fourdivpi
            elif k == Face.Top.value or k == Face.Down.value:
                if q[0] <= 1 and q[0] >= -1 and q[1] <= 1 and q[1] >= -1:
                    face = Face(k)
                q[0] = math.atan(q[0]) * fourdivpi
                q[1] = math.atan(q[1]) * fourdivpi
            if face != Face.Invalid:
                break

        if face == Face.Invalid:
            print("[ImageTransformer] findFaceUV() - Didn't find an intersecting face, shouldn't happen!")
            return False

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
        elif face == Face.Down:
            u = 1 - q[0]
            v = 1 - q[1]
        elif face == Face.Top:
            u = 1 - q[0]
            v = q[1] + 1

        u *= 0.5
        v *= 0.5

        if u >= 1:
            u = 0.9999
        if v >= 1:
            v = 0.9999

        self.m_LUT[index].u = u
        self.m_LUT[index].v = v
        self.m_LUT[index].face = face.value

        return True

    def generatePixelMatrix(self, index):
        face = Face(self.m_LUT[index].face)
        u = self.m_LUT[index].u
        v = self.m_LUT[index].v

        if face == Face.Down or face == Face.Back or face == Face.Top:
            u, v = v, 0.9999 - u

        u_left = u
        u_right = u

        x0 = 0
        w = 0
        duv = 0.0

        if face == Face.Front:
            x0 = self.m_framespecs.sidewidth
            w = self.m_framespecs.centerwidth
            self.m_LUT[index].ix = int(x0 + u * w)
            self.m_LUT[index].iy = int(v * self.m_framespecs.height)
            self.m_LUT[index].alpha = -1.0
        elif face == Face.Back:
            x0 = self.m_framespecs.sidewidth
            w = self.m_framespecs.centerwidth
            self.m_LUT[index].ix = int(max(0, self.m_framespecs.width - 1 - (x0 + u * w)))
            self.m_LUT[index].iy = int(max(0, self.m_framespecs.height - 1 - (v * self.m_framespecs.height)))
            self.m_LUT[index].alpha = -1.0
        elif face == Face.Left:
            w = self.m_framespecs.sidewidth
            duv = self.m_framespecs.blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                self.m_LUT[index].ix = int(u_left * w)
                self.m_LUT[index].iy = int(v * self.m_framespecs.height)
                self.m_LUT[index].alpha = -1.0
            elif u_right >= 0.5 + 2 * duv:
                self.m_LUT[index].ix = int(u_right * w)
                self.m_LUT[index].iy = int(v * self.m_framespecs.height)
                self.m_LUT[index].alpha = -1.0
            else:
                self.m_LUT[index].ix = int(u_left * w)
                self.m_LUT[index].iy = int(v * self.m_framespecs.height)
                self.m_LUT[index].jx = int(u_right * w)
                self.m_LUT[index].jy = self.m_LUT[index].iy
                self.m_LUT[index].alpha = (u_left - 0.5 + 2 * duv) / (2 * duv)
        elif face == Face.Down:
            w = self.m_framespecs.sidewidth
            duv = self.m_framespecs.blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                self.m_LUT[index].ix = int(max(0, self.m_framespecs.width - 1 - (u_left * w)))
                self.m_LUT[index].iy = int(max(0, self.m_framespecs.height - 1 - (v * self.m_framespecs.height)))
                self.m_LUT[index].alpha = -1.0
            elif u_right >= 0.5 + 2 * duv:
                self.m_LUT[index].ix = int(max(0, self.m_framespecs.width - 1 - (u_right * w)))
                self.m_LUT[index].iy = int(max(0, self.m_framespecs.height - 1 - (v * self.m_framespecs.height)))
                self.m_LUT[index].alpha = -1.0
            else:
                self.m_LUT[index].ix = int(max(0, self.m_framespecs.width - 1 - (u_left * w)))
                self.m_LUT[index].iy = int(max(0, self.m_framespecs.height - 1 - (v * self.m_framespecs.height)))
                self.m_LUT[index].jx = int(max(0, self.m_framespecs.width - 1 - (u_right * w)))
                self.m_LUT[index].jy = self.m_LUT[index].iy
                self.m_LUT[index].alpha = (u_left - 0.5 + 2 * duv) / (2 * duv)
        elif face == Face.Right:
            x0 = self.m_framespecs.sidewidth + self.m_framespecs.centerwidth
            w = self.m_framespecs.sidewidth
            duv = self.m_framespecs.blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                self.m_LUT[index].ix = int(x0 + u_left * w)
                self.m_LUT[index].iy = int(v * self.m_framespecs.height)
                self.m_LUT[index].alpha = -1.0
            elif u_right >= 0.5 + 2 * duv:
                self.m_LUT[index].ix = int(x0 + u_right * w)
                self.m_LUT[index].iy = int(v * self.m_framespecs.height)
                self.m_LUT[index].alpha = -1.0
            else:
                self.m_LUT[index].ix = int(x0 + u_left * w)
                self.m_LUT[index].iy = int(v * self.m_framespecs.height)
                self.m_LUT[index].jx = int(x0 + u_right * w)
                self.m_LUT[index].jy = self.m_LUT[index].iy
                self.m_LUT[index].alpha = (u_left - 0.5 + 2 * duv) / (2 * duv)
        elif face == Face.Top:
            x0 = self.m_framespecs.sidewidth + self.m_framespecs.centerwidth
            w = self.m_framespecs.sidewidth
            duv = self.m_framespecs.blendwidth / w
            u_left = 2 * (0.5 - duv) * u
            u_right = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv

            if u_left <= 0.5 - 2 * duv:
                self.m_LUT[index].ix = int(max(0, self.m_framespecs.width - 1 - (x0 + u_left * w)))
                self.m_LUT[index].iy = int(max(0, self.m_framespecs.height - 1 - (v * self.m_framespecs.height)))
                self.m_LUT[index].alpha = -1.0
            elif u_right >= 0.5 + 2 * duv:
                self.m_LUT[index].ix = int(max(0, self.m_framespecs.width - 1 - (x0 + u_right * w)))
                self.m_LUT[index].iy = int(max(0, self.m_framespecs.height - 1 - (v * self.m_framespecs.height)))
                self.m_LUT[index].alpha = -1.0
            else:
                self.m_LUT[index].ix = int(max(0, self.m_framespecs.width - 1 - (x0 + u_left * w)))
                self.m_LUT[index].iy = int(max(0, self.m_framespecs.height - 1 - (v * self.m_framespecs.height)))
                self.m_LUT[index].jx = int(max(0, self.m_framespecs.width - 1 - (x0 + u_right * w)))
                self.m_LUT[index].jy = self.m_LUT[index].iy
                self.m_LUT[index].alpha = (u_left - 0.5 + 2 * duv) / (2 * duv)
        else:
            print(f"[ImageTransformer] Unknown face: {face}")

    def colorBlend(self, c1, c2, alpha):
        alpha_inv = 1 - alpha
        color = alpha_inv * c1 + alpha * c2
        color = np.clip(color, 0, 255).astype(np.uint8)
        return color

    def getPixels(self, index, imageH, imageV, drawDown, drawTop):
        face = Face(self.m_LUT[index].face)
        color = np.array([0, 0, 0], dtype=np.uint8)
        if face == Face.Front:
            color = imageH[self.m_LUT[index].iy, self.m_LUT[index].ix]
        elif face == Face.Left or face == Face.Right:
            if self.m_LUT[index].alpha < 0:
                color = imageH[self.m_LUT[index].iy, self.m_LUT[index].ix]
            else:
                c1 = imageH[self.m_LUT[index].iy, self.m_LUT[index].ix]
                c2 = imageH[self.m_LUT[index].jy, self.m_LUT[index].jx]
                color = self.colorBlend(c1, c2, self.m_LUT[index].alpha)
        elif face == Face.Back:
            color = imageV[self.m_LUT[index].iy, self.m_LUT[index].ix]
        elif drawDown and face == Face.Down:
            if self.m_LUT[index].alpha < 0:
                color = imageV[self.m_LUT[index].iy, self.m_LUT[index].ix]
            else:
                c1 = imageV[self.m_LUT[index].iy, self.m_LUT[index].ix]
                c2 = imageV[self.m_LUT[index].jy, self.m_LUT[index].jx]
                color = self.colorBlend(c1, c2, self.m_LUT[index].alpha)
        elif drawTop and face == Face.Top:
            if self.m_LUT[index].alpha < 0:
                color = imageV[self.m_LUT[index].iy, self.m_LUT[index].ix]
            else:
                c1 = imageV[self.m_LUT[index].iy, self.m_LUT[index].ix]
                c2 = imageV[self.m_LUT[index].jy, self.m_LUT[index].jx]
                color = self.colorBlend(c1, c2, self.m_LUT[index].alpha)
        return color

    def cubemapToSpheric(self, imageH, imageV, imageS, roi=None, drawDown=True, drawTop=True):
        antialias_root = self.m_framespecs.antialias * 2
        itable = 0

        if roi is None:
            roi = (0, 0, self.m_framespecs.equiwidth, self.m_framespecs.equiheight)

        x_start, y_start, width, height = roi
        x_end = x_start + width
        y_end = y_start + height

        if imageS.shape[1] != width or imageS.shape[0] != height:
            imageS = np.zeros((height, width, 3), dtype=np.uint8)

        for j in range(y_start, y_end):
            itable = (j * self.m_framespecs.equiwidth + x_start) * antialias_root
            for i in range(x_start, x_end):
                color = self.getPixels(itable, imageH, imageV, drawDown, drawTop)
                imageS[j - y_start, i - x_start] = color
                itable += antialias_root

        return imageS

    def translate(self, angle, image):
        a = (0 - self.m_framespecs.equiwidth) / (-M_PI - M_PI)
        b = 0 - a * -M_PI
        dx = int(a * angle + b)

        if dx == 0 or dx == image.shape[1]:
            return image

        part_image_l = image[:, :dx].copy()
        part_image_r = image[:, dx:].copy()
        image_out = np.zeros_like(image)
        image_out[:, :part_image_r.shape[1]] = part_image_r
        image_out[:, part_image_r.shape[1]:] = part_image_l

        return image_out
