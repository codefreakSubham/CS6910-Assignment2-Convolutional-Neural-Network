# CS6910-Assignment2
Assignment 2 of the CS6910: Fundamentals of Deep Learning course by Dipra Bhagat (CS21S048) and Subham Das (CS21S058)

## Part A

1. The notebook is well structured and can be run cell by cell. (use **run all** option on jupyter/colab/kaggle. Or you can also run anually cell by cell)

2. Next, the google drive needs to be mounted and the iNaturalist file needs to be unzipped. This part of the code will need to be modified according to the filepath on your local machine.

```python

#Download and unzip iNaturalist zip file onto server,

!wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip
!unzip -q nature_12K.zip

```
3. There are functions defined to build a custom CNN and to prepare the image data generators for training and testing which need to be compiled.

4. There are 3 functions ```train_wandb()```, ```train()``` and ```test()``` for integration of WandB with the training, normal training, validation and testing process.  A sweep config is defined already, whose hyperparameters and values can be modified. The train or test function can be called by the sweep agent.

```python
sweep_config = {
    "name": "Final Sweep(Bayesian)",
    "description": "Tuning hyperparameters",
    'metric': {
      'name': 'val_categorical_accuracy',
      'goal': 'maximize'
  },
    "method": "bayes",
    "project": "CS6910_Assignment2",
    "parameters": {
        "n_filters": {
        "values": [16, 32, 64]
        },
        "filter_multiplier": {
            "values": [0.5, 1, 2]
        },
        "augment_data": {
            "values": [True]
        },
        "dropout": {
            "values": [0.3, 0.5]
        },
        "batch_norm": {
            "values": [False, True]
        },
        "epochs": {
            "values": [5, 10]
        },
        "dense_size": {
            "values": [32, 64, 128]
        },
        "lr": {
            "values": [0.01, 0.001]
        },
        "batch_size": {
            "values": [64, 128, 256]
        },
        "activation": {
            "values": ["relu", "leaky"]
        },
    }
}

# creating the sweep
sweep_id = wandb.sweep(sweep_config, project="Convolutional Neural Networks", entity="cs21s048-cs21s058")
```

The ```train()``` function has been made flexible with the following positional arguments which are initialized with the best parameters found by wandb sweep.

```python
def train(n_filters=256, filter_multiplier=0.5, dropout= 0.3, 
          batch_norm = True, dense_size= 128, act_func= "relu", 
          batch_size=16, augmentation=True):
```

One can also pass custom values while calling the ```train()``` function.

To make predictions on the model, one can call the ```predict()``` function with test data as argument.

```python
predictions = predict(test)
```

5. Also, there is a function which can customise the run names in WandB.

7. All the plots will be saved as a jpg/png file.

For the code to run properly, please **modify the paths** according to your system. Following places modifications may be needed:

```python
!unzip "./nature_12K.zip"  #change path accordingly
```
```python
def train_dataset(augmentation=False, batch_size=64):
    dir_train = './inaturalist_12K/train' #change paths accordingly
    dir_test = './inaturalist_12K/val' #change paths accordingly

```
```python
def predict(test_data):
    
    model = keras.models.load_model("best_model.h5") #change path accordingly
    predictions = model(test[0][0])
    model.evaluate(test_data, batch_size=64)

    return predictions
```

## Part B

1. The notebook is structured such that it can be ran cell by cell

2. Model is made flexible with the following parameters:

```python
def modified_model(pre_trained_model, n_dense, dropout, freeze_before):
```

3. For training the model and fine-tuning with wandb, use `train()` function

```python
# Train Function
def train(config=None):
    
    #Wandb settings
    wandb.init(project="Convolutional Neural Networks", entity="cs21s048-cs21s058")
    config = wandb.config
    wandb.run.name = setWandbName(model_name= config.pre_trained_model, dropout=config.dropout, batch_size=config.batch_size, n_dense=config.n_dense)

    
    train, val, test = generate_data(batch_size= config.batch_size)
    

    new_model = modified_model(pre_trained_model= config.pre_trained_model, n_dense= config.n_dense, dropout= config.dropout, freeze_before= config.freeze_before)
    new_model.compile(optimizer=keras.optimizers.Adam(config.learning_rate), loss='categorical_crossentropy', metrics='categorical_accuracy')
    new_model.fit(
        train,
        batch_size = config.batch_size,
        epochs = config.epochs,
        verbose = 1,
        validation_data= val,
        callbacks = [WandbCallback(),keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )
```

Sweep config used:

```python
sweep_config = {
    'name': 'PartB_final_sweep',
    'method': 'bayes', 
    'metric': {
      'name': 'val_categorical_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
       
       'pre_trained_model' :{
           'values' : ['IV3','IRNV2']#, 'RN50', 'XCP']
       },
        'freeze_before' : {
            'values': [50, 70,100]
        },
        'epochs' : {
            'values': [10]
        },
        'dropout': {
            'values': [0.2, 0.4, 0.6]
        },     
        'batch_size': {
            'values': [32, 64]
        },
        'n_dense':{
            'values': [64, 128, 256]
        },
        'learning_rate':{
            'values': [0.001, 0.0001]
        }
    }
}
#Cr
```