import numpy as np
from numba import jit
from copy import deepcopy
from .kitti_oxts import roty

def _as_scalar_float(v, name="value"):
    """
    Convert scalar / 0-d array / 1-element list / 1-element ndarray to float.
    Raise a clear error for real non-scalar values instead of letting numba crash.
    """
    if v is None:
        return None

    arr = np.asarray(v)

    if arr.dtype == object:
        arr = np.asarray(arr.tolist(), dtype=np.float64)
    else:
        arr = arr.astype(np.float64, copy=False)

    arr = arr.reshape(-1)

    if arr.size != 1:
        raise ValueError(
            f"{name} must be scalar, got shape={arr.shape}, value={v!r}"
        )

    return float(arr[0])

class Box3D:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None, s=None):
        self.x = _as_scalar_float(x, "x")
        self.y = _as_scalar_float(y, "y")
        self.z = _as_scalar_float(z, "z")
        self.h = _as_scalar_float(h, "h")
        self.w = _as_scalar_float(w, "w")
        self.l = _as_scalar_float(l, "l")
        self.ry = _as_scalar_float(ry, "ry")
        self.s = _as_scalar_float(s, "s") if s is not None else None
        self.corners_3d_cam = None

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s)
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.ry}
    
    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def bbox2array_raw(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry])
        else:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s])

    @classmethod
    def array2bbox_raw(cls, data):
        # data format: [h,w,l,x,y,z,theta]
        data = np.asarray(data, dtype=object).reshape(-1)

        if data.size < 7:
            raise ValueError(f"array2bbox_raw expects at least 7 values, got shape={data.shape}, value={data!r}")

        bbox = cls(
        h=data[0],
        w=data[1],
        l=data[2],
        x=data[3],
        y=data[4],
        z=data[5],
        ry=data[6],
        s=data[7] if data.size >= 8 else None,
        )
        return bbox
    
    @classmethod
    def array2bbox(cls, data):
        # data format: [x,y,z,theta,l,w,h]
        data = np.asarray(data, dtype=object).reshape(-1)

        if data.size < 7:
            raise ValueError(f"array2bbox expects at least 7 values, got shape={data.shape}, value={data!r}")

        bbox = cls(
            x=data[0],
            y=data[1],
            z=data[2],
            ry=data[3],
            l=data[4],
            w=data[5],
            h=data[6],
            s=data[7] if data.size >= 8 else None,
        )
        return bbox
    
    @classmethod
    def box2corners3d_camcoord(cls, bbox):
        ''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and 
            convert it to the 8 corners of the 3D box, the box is in the camera coordinate
            with right x, down y, front z
            
            Returns:
                corners_3d: (8,3) array in in rect camera coord

            box corner order is like follows
                    1 -------- 0         top is bottom because y direction is negative
                   /|         /|
                  2 -------- 3 .
                  | |        | |
                  . 5 -------- 4
                  |/         |/
                  6 -------- 7    
            
            rect/ref camera coord:
            right x, down y, front z

            x -> w, z -> l, y -> h
        '''

        # if already computed before, then skip it
        if bbox.corners_3d_cam is not None:
            return bbox.corners_3d_cam

        # compute rotational matrix around yaw axis
        # -1.57 means straight, so there is a rotation here
        ry = _as_scalar_float(bbox.ry, "bbox.ry")

        c = np.cos(ry)
        s = np.sin(ry)

        R = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ], dtype=np.float64)  

        # 3d bounding box dimensions
        x = _as_scalar_float(bbox.x, "bbox.x")
        y = _as_scalar_float(bbox.y, "bbox.y")
        z = _as_scalar_float(bbox.z, "bbox.z")
        l = _as_scalar_float(bbox.l, "bbox.l")
        w = _as_scalar_float(bbox.w, "bbox.w")
        h = _as_scalar_float(bbox.h, "bbox.h")

        # 3d bounding box corners
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
        y_corners = [0,0,0,0,-h,-h,-h,-h];
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + x
        corners_3d[1, :] = corners_3d[1, :] + y
        corners_3d[2, :] = corners_3d[2, :] + z
        corners_3d = np.transpose(corners_3d)
        bbox.corners_3d_cam = corners_3d

        return corners_3d