# Challenge 1
To train your model just run:

`python -m src.train.main.py --config.yaml`

Metrics specified in config.yaml will be created in the `./reports` folder.\
The model will be stored in `./models`\

Change config.yaml to parametrize different stages of the training.
To use a different model, follow the pattern enforced with the interface in src.mode.abc.Classifier.py

There are only a few tests for now, just to show how I usually code them