import os
from collections import defaultdict
from scipy.io import loadmat
import numpy as np
from utils.database import COLMAPDatabase,blob_to_array

DATABASE_PATH = 'sqlite/putin.db'
if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)
db = COLMAPDatabase.connect(DATABASE_PATH)
db.create_tables()

MAT_DIR = 'mat'
mat_files = [f for f in os.listdir('mat') if f[-4:] == '.mat']
class KeyPointsManager():
    def __init__(self):
        self.__kpts = list()
        self.__kpts_len = 0
    def get(self):
        return self.__kpts
    def add(self,data):
        data['id'] = self.__kpts_len + 1
        self.__kpts.append(data)
        self.__kpts_len = self.__kpts_len + 1
    def lenght(self):
        return self.__kpts_len


# add camera and image
cam_param = np.loadtxt('parameter/camera.txt')
for i in range(len(mat_files)):
    cam_id = db.add_camera(model=2, width=256, height=256, params=cam_param)
    file_name = mat_files[i].split('_')[0] + '.jpg'
    db.add_image(file_name,cam_id)


# build keypoint (feature extraction ?)
keypoints_manager = KeyPointsManager()
for image_id,mat_file in enumerate(mat_files[:1],start=1):
    mesh = loadmat(os.path.join(MAT_DIR,mat_file))
    image_projector = defaultdict(dict)
    #select only highest z to mark at keypoint
    for i,vertex in enumerate(mesh['vertices'],start=1):
        x, y, z = np.around(vertex).astype(np.int32)
        if z > 0 and (not y in image_projector[x] or z > image_projector[x][y]['z']):
            image_projector[y][x] = {
                'z': z,
                'v': i #vertex number
            }
    for y in image_projector.keys():
        for x in image_projector[y].keys():
            kpt = image_projector[y][x]
            keypoints_manager.add({
                'image': image_id,
                'x': x,
                'y': y,
                'vertex': kpt['v']
            })
# feature matching
feature_bags = list()
keypoints = keypoints_manager.get()
for from_id in range(1,len(mat_files)+1):
    for to_id in range(i+1,len(mat_files)+1):
        edges = list()
        kpts_from = filter(lambda x: x['image'] == from_id, keypoints)
        kpts_to = filter(lambda x: x['image'] == to_id, keypoints)
        kpts_to = dict([(kpt['vertex'],kpt['id']) for kpt in kpts_to])
        for kpt_from in kpts_from:
            if kpt_from['vertex'] in kpts_to:
                edges.append(
                    (kpt_from['id'], kpts_to[kpt_from['vertex']])
                )
        feature_bags.append({
            'image_from': from_id,
            'image_to': to_id,
            'edge': edges
        })

#mesh = loadmat('mat/0001_mesh.mat')
#mesh['vertices'][0]