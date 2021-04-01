# aeroplane_cleaning

* Final project in UM Data Science.

## what we have done

| Time | Done | Bugs or ToDo |
| :-: | :-: | :-: | 
| 01.25 | Get familiar with Carla <br> Solving installation problem | |
| 02.01 | Create a commercial plane in carla <br> Delete the official map elements,  simulate the airport <br> Spawn a car in airport |  |
| 02.21 | Dynamic weather <br> Manually control | Bug: Low fps |
| 02.28 | Add RGB sensor in async mode <br> Save image to disk <br> Fix bug0221 | Bug: Manual_control failed |
| 03.07 | Fix bug0228 | Todo: Aeroplane segmentation |
| 03.14 | Add Semantic RGB sensor | Failed Todo0307 |
| 03.21 | Cut the plane using Blender, Todo0307 done  | Todo: User defined semantic segmantation |
| 03.23 | Update the Carla to 0.9.11  | Bug: Installation problem |
| 03.26 | Finally fix Bug0323 | Todo0321 |
| 03.27 | Add Lidar and Semantic Lidar | Bug: User define tags failed <br> Bug: User define semantic color palette error |
| 03.31 | Temporarily solved Bug0327 with a lot of re-install and re-build <br> Add sensors together | Bug: Huge frame number difference |
| 04.01 | Switch to sync mode <br> Add toggle reverse KEY in manual control |  Initially solved Bug0331 |



## Some useful link

| Key word | Link |
| :-: | :-: |
| bounding box based on color | https://stackoverflow.com/questions/50051916/bounding-box-on-objects-based-on-color-python |


## Dirty work

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
  * make PythonAPI
  * make Import
  * make LibCarla