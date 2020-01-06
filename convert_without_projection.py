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
print(mat_files)
exit()

#add camera
cam_param = np.asarray([307.2, 128.0, 128.0, 0.0])
for i in range(len(mat_files)):
    cam_id = db.add_camera(model=2, width=256, height=256, params=cam_param)
    file_name = mat_files[i].split(MAT_POSTFIX)[0] + IMAGE_EXTENSION
    db.add_image(file_name,cam_id)

# build keypoint (feature extraction)
kpt_len = 0
for image_id, mat_file in enumerate(mat_files,start=1):
    mesh = loadmat(os.path.join(MAT_DIR,mat_file))
    kpt_len = mesh['vertices'].shape[0]
    keypoints = list()
    #select only highest z to mark at keypoint
    for i,vertex in enumerate(mesh['vertices'],start=1):
        x, y, z = vertex
        keypoints.append( (x,y) )
    # add keypoints to database
    db.add_keypoints(image_id, np.asarray(keypoints))


# feature mathcing
image_matchs = list()
for from_id in range(1,len(mat_files)+1):
    for to_id in range(from_id+1,len(mat_files)+1):
        nrange = np.arange(0,kpt_len)
        pairs = np.c_[nrange,nrange]
        # add match to database
        #db.add_matches(from_id, to_id, np.asarray(pairs))
        db.add_two_view_geometry(from_id, to_id, pairs)

# commit (save) to db 
db.commit()
db.close()
