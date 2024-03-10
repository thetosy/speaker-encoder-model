## Transformer Speaker Encoder

Implemented a transformer-based speaker recognition system using Pytorch.

The model is trained on [LibriSpeech](https://www.openslr.org/12) and evaluated on Equal Error Rate (EER)
### Update model hyperparameter

open `config.py` to edit and adjust the necessary model parameters as needed

### Train a model

Run:

```
python3 encoder.py
```

To begin training the model on train data specified by path `TRAIN_DATA_DIR` in `config.py`

Once training is done, the trained model will be saved to disk, based on the path specified by `SAVED_MODEL_PATH` in `config.py`.

### Evaluate the model

Once the model has been trained and saved to disk

Run:

```
python3 eval.py
```

to evaluate the Equal Error Rate (EER) of the model on test data.


Adjust `NUM_EVAL_TRIPLETS` in `config.py` to control how many positive and negative trials you want to evaluate.

When the evaluation job finishes, the EER will be printed to screen

## References
1. [Speaker Recognition](https://www.udemy.com/course/speaker-recognition/?couponCode=ACCAGE0923), Wang Q.
