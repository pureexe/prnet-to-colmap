import os
from collections import defaultdict
from scipy.io import loadmat
import numpy as np
from utils.database import COLMAPDatabase,blob_to_array

DATABASE_PATH = 'sqlite/putin_without_projection.db'
MAT_DIR = 'mat'
MAT_POSTFIX = '_mesh'
IMAGE_EXTENSION = '.jpg'
if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)
db = COLMAPDatabase.connect(DATABASE_PATH)
db.create_tables()

mat_files = [f for f in os.listdir('mat') if f[-4:] == '.mat']

#add camera
cam_param = np.asarray([307.2, 128.0, 128.0, 0.0])
for i in range(len(mat_files)):
    cam_id = db.add_camera(model=2, width=256, height=256, params=cam_param)
    file_name = mat_files[i].split(MAT_POSTFIX)[0] + IMAGE_EXTENSION
    db.add_image(file_name,cam_id)

# build keypoint (feature extraction)
images_keypoints = dict()
images_keypoints_map = defaultdict(dict)
for image_id, mat_file in enumerate(mat_files,start=1):
    mesh = loadmat(os.path.join(MAT_DIR,mat_file))
    image_projector = defaultdict(dict)
    #select only highest z to mark at keypoint
    for i,vertex in enumerate(mesh['vertices'],start=1):
        x, y, z = np.around(vertex).astype(np.int32)
        if z > 0 and (not y in image_projector[x] or z > image_projector[x][y]['z']):
            image_projector[x][y] = {
                'z': z,
                'v': i #vertex number
            }
    images_keypoints[image_id] = list()
    len_kpt = 0
    for x in image_projector.keys():
        for y in image_projector[x].keys():
            kpt = image_projector[x][y]
            images_keypoints[image_id].append( (x,y) )
            images_keypoints_map[image_id][kpt['v']] = len_kpt
            len_kpt = len_kpt + 1
    # add keypoints to database
    db.add_keypoints(image_id, np.asarray(images_keypoints[image_id]))


# feature mathcing
image_matchs = list()
for from_id in range(1,len(mat_files)+1):
    for to_id in range(from_id+1,len(mat_files)+1):
        pairs = list()
        for kpt_index,kpt_marker in enumerate(images_keypoints_map[from_id]):
            if kpt_marker in images_keypoints_map[to_id]:
                pairs.append(
                    (kpt_index, images_keypoints_map[to_id][kpt_marker])
                )
        # add match to database
        #db.add_matches(from_id, to_id, np.asarray(pairs))
        db.add_two_view_geometry(from_id, to_id, np.asarray(pairs))

# commit (save) to db 
db.commit()
db.close()
