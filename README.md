# Privacy-in-Medical-Imaging

- Download the Datasets using [Download Kaggle Dataset.ipynb](https://github.com/cybr17crwlr/Privacy-in-Medical-Imaging/blob/main/datasets/Download%20Kaggle%20Dataset.ipynb)

- After that you can run any of the python scripts in the three folders for different kinds of models

| Folder Name | Type of Training                                              |
| ----------- | ------------------------------------------------------------- |
| Augment     | Dataset with only Data Augmentation in training pipeline      |
| Finetune    | Dataset with only Fine tuning training pipeline               |
| Aug+FT      | Dataset with only Data Aug and finetuning in training pipeline|

The naming structure means `<feature extractor>_<input image size>_<training type>.py`

Currently there is only one featur extractor [EfficientNetv2B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet_v2/EfficientNetV2B0) labelled as `EFFNET`.

The training types are `BS` for basestats training without differential privacy and `DP` is differentially private training.

The training results will be stored in the `logs` folder and the training checkpoints will be stored in the `checkpoints` folder.

For any other queries please contact:

[Apoorva Kumar](https://www.linkedin.com/in/kr17apoorva) - [kr17apoorva@gmail.com](mailto:kr17apoorva@gmail.com)
