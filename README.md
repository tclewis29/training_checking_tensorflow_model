# Tensorflow model training
This set of scripts will enable you to go from a csv of labelled data to a trained tensorflow model and check its performance on validation and test data. The model is created with the end use in mind, which is taking long passive acoustic monitoring recordings and detecting target species signal. 

## Features
- It will accommodate as many classes as you have, it will automatially detect the number of classes so you don't need to worry about assigning them. 
- Can be run from start to finish or in chunks
- Requires access to training recordings to create training dataset 
- Will display performance metrics as well as save results to csvs

## Data extraction
To train the model first you need your labelled training data in the correct format. For these models you need the spectrograms to be converted into arrays. These will then be stored as pickle files for use downstream.
```sh
from preprocessing import AudioProcessor

# create an instance of AudioProcessor
processor = AudioProcessor(audio_path = "E:/2020 PhD/Abundance/", # the path to where your audio files are stored
                            save_pickled_data = "./", # where you want to save the training pickle file
                            clip_dur = 1, # length of spectrogram used in the model
                            file_name = "my_training_csv_name", # doesn't need the .csv suffix
                            downsample_rate = 22050, hann_win = 1024, h_len = 256, nmels = 128, fmin = 500, fmax = 9000) 

# use the methods of the AudioProcessor class
X_Calls, Y_Calls = processor.get_data(training_files = "./training_dataset_split_1500.csv") # csv has to have columns: sound.files, start, end, Annotation
```
#### Example Output
```sh
100%|███████████████████████████████████████████████████████████████████████████████| 53177/53177 [44:29<00:00, 19.92it/s]
Nailed it
```
## Training the Tensorflow model
We are creating a custom Tensorflow model using the data extracted above. These class of functions also contains the options to measure performance across all classes or just focal species classes. 

The function will automatically detect the number of classes in your training dataset and set the final number of dense layers in the model to that. It will also output a file named ```class_labels.csv```, which is the reference for the numerical classes used within the model as tensorflow models cannot handle catagorical labels.
```sh
from custom_model import CustomModel

# create an instance of AudioModelTrainer
trainer = CustomModel(save_pickled_data = "./", 
                        INPUT_SHAPE = (128, 87, 3), #  shape of the arrays, this should stay like this unless it has been changed when doing data extraction
                        BS = 32 , # Batch size
                        EPOCHS = 100, # The number of epochs to use. The training uses early stopping so keeping epochs high is advised
                        training_file="my_training_csv_name", # is the same as file_name 
                        model_name = "my_new_model_name") 

# training the model
model = trainer.train_model()

# get validation performance metrics
trainer.performance(model)

# focal species metrics
trainer.focal_performance(focal_species = ['GGM','SCM'])
```
#### Example Output
```sh
Original Data distribution: {'Amazona spp.': 4936, 'Chainsaw': 1048, 'Chicken': 1690, 'Clay coloured thrush': 2509}
Label to Number mapping: {'Amazona spp.': 0, 'Chainsaw': 1, 'SCM': 2, 'Chicken': 3, 'Clay coloured thrush': 4}
Number to label mapping: {0: 'Amazona spp.', 1: 'Chainsaw', 2: 'SCM', 3: 'Chicken', 4: 'Clay coloured thrush'}
Model: "sequential"
_________________________________________________________________________________
 Layer (type)                                   Output Shape              Param #
=================================================================================
 conv2d (Conv2D)                                (None, 128, 87, 32)     896
 max_pooling2d (MaxPooling2D)                    (None, 64, 43, 32)      0
 batch_normalization (BatchNormalization)       (None, 64, 43, 32)      128
 conv2d_1 (Conv2D)                              (None, 64, 43, 32)      9248
 max_pooling2d_1 (MaxPooling2D)                 (None, 32, 21, 32)      0
 batch_normalization_1 (BatchNormalization)     (None, 32, 21, 32)      128
 conv2d_2 (Conv2D)                              (None, 32, 21, 64)      18496
 max_pooling2d_2 (MaxPooling2D)                 (None, 16, 10, 64)      0
 batch_normalization_2 (BatchNormalization)     (None, 16, 10, 64)      256
 conv2d_3 (Conv2D)                              (None, 16, 10, 64)      36928
 max_pooling2d_3 (MaxPooling2D)                 (None, 8, 5, 64)        0
 batch_normalization_3 (BatchNormalization)     (None, 8, 5, 64)        256
 conv2d_4 (Conv2D)                              (None, 8, 5, 64)        36928
 max_pooling2d_4 (MaxPooling2D)                 (None, 4, 2, 64)        0
 batch_normalization_4 (BatchNormalization)     (None, 4, 2, 64)        256
 flatten (Flatten)                              (None, 512)             0
 dense (Dense)                                  (None, 128)             65664
 dense_1 (Dense)                                (None, 4)               1677
=================================================================================
Total params: 170,861
Trainable params: 170,349
Non-trainable params: 512
_________________________________________________________________________________
Epoch 1/100
1055/1055 [==============================] - 116s 108ms/step - loss: 2.3439 - accuracy: 0.2803 - val_loss: 2.1220 - val_accuracy: 0.3331
Epoch 2/100
1055/1055 [==============================] - 113s 107ms/step - loss: 2.0108 - accuracy: 0.3582 - val_loss: 1.9781 - val_accuracy: 0.3691
Epoch 3/100
1055/1055 [==============================] - 113s 107ms/step - loss: 1.8905 - accuracy: 0.3909 - val_loss: 1.8936 - val_accuracy: 0.3894
Epoch 4/100
1055/1055 [==============================] - 113s 107ms/step - loss: 1.8068 - accuracy: 0.4141 - val_loss: 1.8314 - val_accuracy: 0.4083
Epoch 5/100
1055/1055 [==============================] - 111s 105ms/step - loss: 1.7409 - accuracy: 0.4334 - val_loss: 1.7802 - val_accuracy: 0.4229
Epoch 6/100
1055/1055 [==============================] - 113s 107ms/step - loss: 1.6814 - accuracy: 0.4528 - val_loss: 1.7383 - val_accuracy: 0.4344
Epoch 7/100
1055/1055 [==============================] - 111s 105ms/step - loss: 1.6344 - accuracy: 0.4696 - val_loss: 1.7034 - val_accuracy: 0.4478
```
When training is complete two plots will be displayed. One will be the training and validation accurary and the second is the training and validation loss. These are saved automatically. You need to close each plot so the process can continue.

The next stage will automoatically evaluate the performance on a validation dataset that is taken from the inital training dataset. Once it has done this it will display a confusion matrix using the normalised accuracy. It will also print the raw true positive, true negative, false positives and false negatives as well as the precision and recall.
```sh

```
## Predict on new data
These functions take a set of input recordings, split them into segments (1 sec duration), classifies them using the specified model and creates an output csv with probabilities for each class as well as a overall label for each segment.
```sh
from predict import AudioPredictor

# create an instance of AudioPredictor
predictor = AudioPredictor(test = "Yes", # If 'Yes' predict over a randomly selected sample. 'No' will predict over all recordings in 'audio_path'
                            audio_path = "E:/2020 PhD/Abundance/", 
                            class_labels = "class_labels.csv", # this file is created when training the model 
                            INPUT_SHAPE = (128, 87, 3),
                            downsample_rate = 22050, hann_win = 1024, h_len = 256, nmels = 128, fmin = 500, fmax = 9000) # ensure these are the same as above

# If test is set to yes we need to define the test variables
predictor.get_test_info(sample_size= 5, test_csv = "./test_soundfiles.csv")

# predict
results = predictor.predict_and_save(model_name="my_model.h5") # change this to your model 
```
#### Example Output
```sh
Generating spectrograms...
100%|██████████████████████████████████████████████████████████████████████| 20/20 [03:47<00:00, 11.40s/it] 
Predicting...
1125/1125 [==============================] - 32s 28ms/step
Done
```
## Evaluate 
If you have manually labelled reference data for the files that have been predicted on above then you can use this function to evaluate the performance of the model of new out-of-sample test data. In many cases it is important to do this as performance can vary significantly, especially if recordigns come from highly heterogenious ecosystems such as tropical rainforests and/or span long time periods.
```sh
from performance_checker import AudioChecker

# create an instance of AudioChecker
checker = AudioChecker(txt_files_dir = "C:/Users/bi1tl/Documents/Manually_labelled", 
                        input_data = "predictions.csv")

# measure performance with focal species
checker.run(focal_species=['GGM','SCM']) 

# measure performance across all species/classes
checker.run() 
```
#### Example Output
```sh
Number of unique sound files: 20

Raw metrics for GGM
True Positives: 2
True Negatives: 10450
False Positives: 138
False Negatives: 210
For GGM - Precision: 0.014, Recall: 0.009

Raw metrics for SCM
True Positives: 2
True Negatives: 10258
False Positives: 509
False Negatives: 31
For SCM - Precision: 0.004, Recall: 0.061
Done
```

