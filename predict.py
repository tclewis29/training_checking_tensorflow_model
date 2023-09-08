import os
import numpy as np
import librosa
from tqdm import tqdm
from datetime import datetime
import math
import soundfile as sf
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

class AudioPredictor:
    def __init__(self, test, audio_path, class_labels, INPUT_SHAPE, downsample_rate, 
                 hann_win, h_len, nmels, fmin, fmax):
        self.test = test
        self.audio_path = audio_path
        self.downsample_rate = downsample_rate
        self.hann_win = hann_win
        self.h_len = h_len
        self.nmels = nmels
        self.fmin = fmin
        self.fmax = fmax
        self.INPUT_SHAPE = INPUT_SHAPE
        self.class_labels = class_labels

    def load_class_labels(self):
        self.class_labels = pd.read_csv(self.class_labels)

    def read_audio(self, path, start_time):
        y, _ = librosa.load(path, offset=start_time, duration = 1, sr = self.downsample_rate)
        return y

    def audio_to_melspectrogram(self, y):
        S = librosa.feature.melspectrogram(y = y, sr = self.downsample_rate, n_fft = self.hann_win, hop_length = self.h_len, n_mels = self.nmels, fmin = self.fmin, fmax = self.fmax)
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = (image_np - np.min(image_np))
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps=1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled

        return np.stack([S1,S1**3,S1**5], axis=2)

    def read_as_melspectrogram(self, path, start_time):
        y = self.read_audio(path, start_time)
        mels = self.audio_to_melspectrogram(y)
        return mels

    def get_audio_duration(self, filename):
        f = sf.SoundFile(filename)
        return len(f) / f.samplerate
    
    def get_test_info(self, sample_size, test_csv):
        self.sample_size = sample_size
        self.test_csv = test_csv

    def get_input_data(self):
        # get a list of all the files
        if self.test == "Yes":
            # Read the CSV file into a DataFrame
            df = pd.read_csv(self.test_csv)
            
            # Sample randomly from the entire DataFrame
            sample_df = df.sample(min(len(df), self.sample_size))
            
            # Ensure the file paths are unique
            files = sample_df['sound.files'].unique()
        else:
            files = [f for f in os.listdir(self.audio_path) if os.path.isfile(os.path.join(self.audio_path, f)) and f.endswith(('.wav', '.WAV'))]

        # for storage
        X = []
        Y = []

        # get the spectrograms
        print('Generating spectrograms...')
        for file in tqdm(files):
            if os.path.exists(self.audio_path + file):
                duration = self.get_audio_duration(self.audio_path + file)
                for i in range(0, math.floor(duration)):
                    x = self.read_as_melspectrogram(self.audio_path + file, i)
                    X.append(x)
                    Y.append(file)
            else:
                print("Cannot find", file)

        X = np.array(X) 
        Y = np.array(Y)    

        return X, Y

    
    def load_model(self, model_name):
        
        # Load your TensorFlow model here
        model = tf.keras.models.load_model(model_name)
        
        return model
    
    def predict_and_save(self, model_name):

        # get the spectrograms
        spectrograms, soundfile = self.get_input_data()
        
        # Check the type of model_name
        if isinstance(model_name, str):
            # If it's a string, assume it's a path and load the model from that path
            model = self.load_model(model_name)
        elif isinstance(model_name, Model):
            # If model_name is already a Model instance, use it as is
            model = model_name
        else:
            raise ValueError("Invalid type for model_name. It should either be a string (path to the model) or a Model instance.")
        
        # Get the number of output classes from the model
        output_layer = model.layers[-1]
        num_classes = output_layer.units
        
        # Load the class mapping
        class_map = pd.read_csv(self.class_labels)
        class_names = class_map['Label'].tolist()

        # Verify if number of classes in model and CSV match
        if num_classes != len(class_names):
            raise ValueError("Number of classes in the model and CSV file do not match.")

        # Initialize empty dataframe with dynamic columns
        columns = ['sound_files', 'start', 'end'] + class_names + ['label']
        df = pd.DataFrame(columns=columns)

        # Initialize empty dataframe with dynamic columns
        columns = ['sound_files', 'start', 'end'] + class_names + ['label']
        df = pd.DataFrame(columns=columns)

        # predict the whole set of spectrograms
        print('Predicting...')
        prediction = model.predict(spectrograms)

        # Initialize variables
        prev_sound_file = soundfile[0]
        time_counter = 0

        # Iterate over spectrograms
        for i, preds in enumerate(prediction):
            # Get sound file for current spectrogram
            sound_file = soundfile[i]

            # If sound file changes, reset time_counter
            if sound_file != prev_sound_file:
                time_counter = 0
            
            # insert start, end, prediction into dataframe
            start_time = time_counter
            end_time = time_counter + 1

            # Creating a row dictionary to handle dynamic columns
            row_dict = {
                'sound_files': sound_file,
                'start': start_time,
                'end': end_time,
                'label': class_map.loc[np.argmax(preds), 'Label']
            }
            for j, class_name in enumerate(class_names):
                row_dict[class_name] = preds[j]

            df.loc[i] = row_dict

            # Update variables for next iteration
            prev_sound_file = sound_file
            time_counter += 1

        # Create the file name by appending the timestamp
        filename = f'predictions.csv'

        # Write to csv with the new file name
        df.to_csv(filename, index=False)
        print('Done')
        return df
