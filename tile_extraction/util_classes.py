import cv2

class Contour_Checking_fn(object):
    # Defining __call__ method
    def __call__(self, pt):
        raise NotImplementedError


class isInContourV1(Contour_Checking_fn):
    def __init__(self, contour):
        self.cont = contour

    def __call__(self, pt):
        return 1 if cv2.pointPolygonTest(self.cont, pt, False) >= 0 else 0


class isInContourV2(Contour_Checking_fn):
    def __init__(self, contour, patch_size):
        self.cont = contour
        self.patch_size = patch_size

    def __call__(self, pt):
        return 1 if cv2.pointPolygonTest(self.cont, (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2),
                                         False) >= 0 else 0


# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(Contour_Checking_fn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (int(pt[0] + self.patch_size // 2), int(pt[1] + self.patch_size // 2))
        if self.shift > 0:
            all_points = [(center[0] - self.shift, center[1] - self.shift),
                          (center[0] + self.shift, center[1] + self.shift),
                          (center[0] + self.shift, center[1] - self.shift),
                          (center[0] - self.shift, center[1] + self.shift)
                          ]
        else:
            all_points = [center]

        for points in all_points:
            if cv2.pointPolygonTest(self.cont, points, False) >= 0:
                return 1
        return 0


# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class isInContourV3_Hard(Contour_Checking_fn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (int(pt[0] + self.patch_size // 2), int(pt[1] + self.patch_size // 2))
        if self.shift > 0:
            all_points = [(center[0] - self.shift, center[1] - self.shift),
                          (center[0] + self.shift, center[1] + self.shift),
                          (center[0] + self.shift, center[1] - self.shift),
                          (center[0] - self.shift, center[1] + self.shift)
                          ]
        else:
            all_points = [center]

        for points in all_points:
            if cv2.pointPolygonTest(self.cont, points, False) < 0:
                return 0
        return 1
