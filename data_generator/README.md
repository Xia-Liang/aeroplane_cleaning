Windows 10

# basic
* `config*` file to import necessary lib abd add basic attribute

# data collection
* `sync_rgb.py` collect sem lidar data and rgb 
* `trainset_sem_lidar.py` collect **training set** data 
* `testset.py` collect test set data, both lidar and semlidar

# data visualization
* `open3d_lidar.py` only for vis 
* `lidar2camera` for vis and saving

# data evaluation
* `test_realtime.py` load the trained model, print how many points in each tag (prediction and real) when running the client