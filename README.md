# aeroplane_cleaning

* Final project in UM Data Science.
* Measure distance to each part of an aircraft
* Use Carla and UE4 to generate data
* Use PointNet to do 3D segmentation
* Get the distance and vis it in UE4

# File Structure

* Put `CPcut` or `Airplane` folder in `carla\Import` and rebuild
  * [Carla add semantic tags](https://carla.readthedocs.io/en/0.9.11/tuto_D_create_semantic_tags/)
  * [Carla add new props](https://carla.readthedocs.io/en/0.9.11/tuto_A_add_props/)
* `data_generator` to get data using Carla on windows10
* `data_preprocess_training` to deal with raw data and train segmentation model on Linux


# what we have done

| Time | Done or Conclusion | Bugs or ToDo (red for remaining) |
| :-: | :-: | :-: | 
| ~01.25 | Get familiar with Carla <br> Solving installation problem | |
| ~02.01 | Create a commercial plane in carla <br> Delete the official map elements,  simulate the airport <br> Spawn a car in airport |  |
| ~02.21 | Dynamic weather <br> Manually control | Bug: Low fps |
| ~02.28 | Add RGB sensor in async mode <br> Save image to disk <br> Fix bug0221, [UE4 settings](https://carla.readthedocs.io/en/0.9.11/build_faq/) or Don't save data per frame | Bug: Manual_control failed |
| ~03.07 | Fix bug0228 | Todo: Aeroplane segmentation |
| ~03.14 | Add Semantic RGB sensor | Failed Todo0307 |
| ~03.21 | Done Todo0307: Cut the plane using Blender  | Todo: User defined semantic segmantation |
| ~03.23 | Update the Carla to 0.9.11  | Bug: Installation problem |
| ~03.26 | Finally fix Bug0323 | Todo0321 |
| ~03.27 | Add Lidar and Semantic Lidar | Bug: User define tags failed <br> <font color=red>Bug: User define semantic color palette error</font> |
| ~03.31 | Temporarily solved Bug0327 with a lot of re-install and re-build <br> Add sensors together | Bug: Huge frame number difference |
| ~04.01 | Switch to sync mode <br> Add toggle reverse KEY in manual control |  Initially solved Bug0331 |
| ~04.02 | Can't control car with open3d vis <br> Solve Bug0331 <br> Done Todo0402: Tags are right, Successfully save lidar data | Test if added tags are right (new empty map) <br> Save lidar data to disk <br> <font color=red>Switch sensor in Pygame window, not necessary</font> |
| ~04.08 | Solving env of ubuntu | |
| ~04.15 | Data prepare and preprocessing <br> Change saving data's naming rules | |
| ~04.19 | Data Training <br> Re-construct the file | <font color=red> Project lidar point to pygame OR Some other visualization</font> |
| ~04.22 | Training model reach to 94% acc using [PointNet](https://arxiv.org/abs/1612.00593) | |
| ~04.23 | | 3D projection <br> Real-time projection <br> Real-time segmentation <br> Distance calculation |
| ~04.25 | Done Todo0423: 3D projection, Real-time saving | |
| ~04.29 | Get test dataset, Failed | <font color=red>Failed</font> |
| ~05.06 | Re-cut the plane to get data more uniformed <br> Another attempt |


# what we can't do and reason

* Real-time bounding box
  * In yolo3, there exists a way of generating bounding box, it uses point clustering method
  * But in this work, the plane model is connected, clustering is not a good idea to do segmentation
  * So we use [PointNet](https://arxiv.org/abs/1612.00593)
  * After we get the tags of each point, we still can't generate good real-time bounding box. Since lidar is set to be 10Hz and the simulator has unstable frames 30~60Hz.

# Some useful link

| Key word | Link |
| :-: | :-: |
| bounding box based on color | https://stackoverflow.com/questions/50051916/bounding-box-on-objects-based-on-color-python |


# Dirty work

* carla install egg file
  * download the releases file
  * extract to PythonAPI/carla/dist
  * https://github.com/carla-simulator/carla/issues/1466
  * https://github.com/carla-simulator/carla/issues/2463#issuecomment-608938363
  * pip install -e C:\Carla0.9.11\PythonAPI\carla\dist\carla-0.9.11-py3.7-win-amd64

* make PythonAPI error
  * https://github.com/carla-simulator/carla/issues/1372
  * py -3.7 -m pip install -e C:\carla0.9.11\PythonAPI\carla\dist\carla-0.9.11-py3.7-win-amd64

* make
  * re-make when you changed .cpp .h file in carla (user defined tags and color palette)
  * make clean
  * remember to add .egg file
  * make PythonAPI
  * make Import
  * make LibCarla

* <font color=red> Solved, no bug </font> System bug in `carla.ColorConverter.CityScapesPalette`
  * Things go right when add user defined tags, see [Carla add semantic tags](https://carla.readthedocs.io/en/0.9.11/tuto_D_create_semantic_tags/) and [Carla add new props](https://carla.readthedocs.io/en/0.9.11/tuto_A_add_props/)
  * But failed to converte the color in version 0.9.11
  * Could get right tags according to `abandoned\check_defined_tag.py`
  * Can't to convert image sem color, see [Semantic segmentation camera](https://carla.readthedocs.io/en/0.9.11/ref_sensors/#semantic-segmentation-camera),  the tag information encoded in the red channel
