Linux Ubuntu18.04

# data saving
* You should have the same folder structure in `data`
    * `airplaneCategory.txt` should be add in `data` folder
* `data_preprocess.py` deal with raw data collect by carlaAPI (see details in `data_generator`)
    * `x y z tag` for each row of numpy bin file

# training
* `dataset.py` to generate data
* `model.py` define the model
* `train_segmentaion.py`
* `test_evaluation.py` print the mIOU for each class using model
