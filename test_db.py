from utils.database import COLMAPDatabase,blob_to_array
import numpy as np
import json

"""
c.execute("SELECT params FROM cameras LIMIT 1")
data = c.fetchall()
matches_data = blob_to_array(data[0][0],np.float64)
print(matches_data)
"""
db = COLMAPDatabase.connect('sqlite/putin.db')
c = db.cursor()
c.execute("DELETE FROM cameras")
params = np.asarray([307.2, 128.0, 128.0, 0.0])
for i in range(1,228):
    db.add_camera(2, 256, 256, params,prior_focal_length=False, camera_id=i)
db.commit()
db.close()