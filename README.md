## Transformer Speaker Encoder

Implemented a transformer-based speaker recognition system using Pytorch.

The model is trained on [LibriSpeech](https://www.openslr.org/12) and evaluated on Equal Error Rate (EER)

### Train a model

you can start training a speaker encoder model by running:

```
python3 encoder.py
```

Update `config.py` to adjust parameters

Once training is done, the trained model will be saved to disk, based on the path specified by `SAVED_MODEL_PATH` in `config.py`.

### Evaluate the model

Once the model has been trained and saved to disk

Run:

```
python3 evaluation.py
```
to evaluate the Equal Error Rate (EER) of the model on test data.


Adjust `NUM_EVAL_TRIPLETS` in `config.py` to control how many positive and negative trials you want to evaluate.

When the evaluation job finishes, the EER will be printed to screen

## References
1. [Speaker Recognition](https://www.udemy.com/course/speaker-recognition/?couponCode=ACCAGE0923), Wang Q.
