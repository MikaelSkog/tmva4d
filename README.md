# TMVA4D
The source code provided here allows for training and evaluating TMVA4D, a CNN architecture for semantic segmentation of radar data. TMVA4D takes radar heat maps in multiple views as input to predict segmentation masks in the elevation-azimuth view.

## Original Paper by Ouaknine et al.
[Multi-View Radar Semantic Segmentation](https://arxiv.org/abs/2103.16214), ICCV 2021.

[Arthur Ouaknine](https://arthurouaknine.github.io/), [Alasdair Newson](https://sites.google.com/site/alasdairnewson/), [Patrick Pérez](https://ptrckprz.github.io/), [Florence Tupin](https://perso.telecom-paristech.fr/tupin/), [Julien Rebut](https://scholar.google.com/citations?user=BJcQNcoAAAAJ&hl=fr)

Code and pre-trained models: <https://github.com/valeoai/MVRSS>

## Installation with Docker

It is strongly recommended that you use Docker with the provided [Dockerfile](./Dockerfile) containing all the dependencies.

0. Clone the repo:
```bash
$ git clone https://github.com/MikaelSkog/tmva4d.git
```

1. Create the Docker image:
```bash
$ cd tmva4d/
$ docker build . -t "tmva4d:Dockerfile"
```

**Note**: The CARRADA dataset used for train and test is considered as already downloaded by default. If it is not the case, you can uncomment the corresponding command lines in the [Dockerfile](./Dockerfile) or follow the guidelines of the dedicated [repository](https://github.com/valeoai/carrada_dataset).

2. Run a container and join an interactive session. Note that the option `-v /host_path:/local_path` is used to mount a volume (corresponding to a shared memory space) between the host machine and the Docker container and to avoid copying data (logs and datasets). You will be able to run the code on this session:
```bash
$ docker run -d --ipc=host -it -v /host_machine_path/datasets:/home/datasets_local -v /host_machine_path/logs:/home/logs --name tmva4d --gpus all tmva4d:Dockerfile sleep infinity
$ docker exec -it tmva4d bash
```


## Installation without Docker

You can either use Docker with the provided [Dockerfile](./Dockerfile) containing all the dependencies, or follow these steps.

0. Clone the repo:
```bash
$ git clone https://github.com/MikaelSkog/tmva4d.git
```

1. Install this repository using pip:
```bash
$ cd tmva4d/
$ pip install -e .
```
With this, you can edit the tmva4d code on the fly and import functions and classes of tmva4d into other projects as well.

2. Install all the dependencies using pip and conda, please take a look at the [Dockerfile](./Dockerfile) for the list and versions of the dependencies.

3. Optional. To uninstall this package, run:
```bash
$ pip uninstall tmva4d
```

You can look at the [Dockerfile](./Dockerfile) if you are uncertain about the steps to install this project.


## Running the code

In any case, it is **mandatory** to specify beforehand both the path where the Dataset4D dataset is located and the path to store the logs and models. Example: I put the Dataset4D folder in /home/datasets_local, the path I should specify is /home/datasets_local. The same way if I store my logs in /home/logs. Please run the following command lines while adapting the paths to your settings:

```bash
$ cd tmva4d/tmva4d/utils/
$ python set_paths.py --dataset4d <path_to_dataset_parent_dir> --logs <path_to_logs_parent_dir>/logs
```

### Training

In order to train a model, a JSON configuration file should be set. The configuration file corresponding to the selected parameters to train the TMVA-Net architecture is provided here: `tmva4d/tmva4d/config_files/tmva4d.json`. To train the TMVA-Net architecture, please run the following command lines:

```bash
$ cd tmva4d/tmva4d/
$ python train.py --cfg config_files/tmva4d.json
```

### Testing

To test a recorded model, you should specify the path to the configuration file recorded in your log folder during training. For example, if you want to test a model and your log path has been set to `/home/logs`, you should specify the following path: `/home/logs/dataset4d/tmvanet/name_of_the_model/config.json`. This way, you should execute the following command lines:

```bash
$ cd tmva4d/tmva4d/
$ python test.py --cfg <path_to_dataset_parent_dir>/tmva4d/name_of_the_model/config.json
```
Note: the current implementation of this script will generate qualitative results in your log folder. You can disable this behavior by setting `get_quali=False` in the parameters of the `predict()` method of the `Tester()` class.


## Acknowledgements
- The paper is under review, special thanks will be indicated after the final results
- The Soft Dice loss is based on the code from <https://github.com/kornia/kornia/blob/master/kornia/losses/dice.py>

## License

The tmva4d repo is released under the Apache 2.0 license.
