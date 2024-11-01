# NutNet
The implementation of CCS 2024 "I Don't Know You, But I Can Catch You: Real-Time Defense against Diverse Adversarial Patches for Object Detectors"

Demos can be found here: [https://sites.google.com/view/nutnet](https://sites.google.com/view/nutnet).

The full version of the paper with an appendix can be found here: [https://arxiv.org/abs/2406.10285](https://arxiv.org/abs/2406.10285).

## Files Description

- [model_weights/](model_weights/) - pretrained weights of the models
- [ae_train_process/](ae_train_process/) - some original and reconstructed images when training the autoencoder
- [ae_weights/](ae_weights/) - the trained autoencoder's weight
- [Dataset/](Dataset/) - the dataset we use for evaluation
- [test_map/](test_map/) - the detection results on the validation dataset

- [ae_trainer.ipynb](ae_trainer.ipynb) - train and test the autoencoder
- [test_map_defense.ipynb](test_map_defense.ipynb) - evaluate the AP (average precision) of the detection model with defense
- [test_map_vanilia.ipynb](test_map_vanilia.ipynb) - evaluate the AP (average precision) of the detection model without defense
- [patchFilter.py](patchFilter.py) - combine the defense and the detection model for convenience
