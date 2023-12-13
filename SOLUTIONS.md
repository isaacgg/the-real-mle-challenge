# Challenge 1
To train your model just run:

`python -m src.train.main.py --config.yaml`

Metrics specified in config.yaml will be created in the `./reports` folder.\
The model will be stored in `./models`\ although all these can be customized changing the config.yaml file.

config.yaml allows you to parametrize different stages of the training.
To use a different model, follow the pattern enforced with the interface in src.mode.abc.Classifier.py

There are only a few tests for now.

# Challenge 2
To run the web api locally use:
`python -m src.api.main.py --config.yaml`


# Challenge 3
Build the image with:
`docker build -t api:latest .`
Run the container with:
`docker run -p 0.0.0.0:8008:8008 api:latest`


## Some notes:
Ideally, I would have enclosed the api in its own class, with the inference class as a dependency, thus allowing me to test it easily. \\
Also, the fitting process should be decoupled from the model itself, but because in this case it's using a very simple model and this is just a test and I didn't want to spend much more time here, I leave it like this.\\ 
I thought it may be seeing as a little over-engineering.
Because of the same reason, I haven't register the training process and model in MlFLow.