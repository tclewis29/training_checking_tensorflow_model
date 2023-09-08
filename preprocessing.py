import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import pickle

class AudioProcessor:
    def __init__(self, audio_path, save_pickled_data, file_name,
                 clip_dur, downsample_rate, hann_win, h_len, nmels, fmin, fmax):
        self.audio_path = audio_path
        self.file_name = file_name
        self.save_pickled_data = save_pickled_data
        self.clip_dur = clip_dur
        self.downsample_rate = downsample_rate
        self.hann_win = hann_win
        self.h_len = h_len
        self.nmels = nmels
        self.fmin = fmin
        self.fmax = fmax

    def read_audio(self, path, start_time, end_time):
        if self.clip_dur > end_time - start_time:
            mid_point = start_time + (end_time - start_time) / 2
            start_point = mid_point - (self.clip_dur / 2)

            if start_point - self.clip_dur <= 0:
              start_point = 0      
        else: 
            start_point = start_time

        y, sr = librosa.load(path, sr = self.downsample_rate, offset = start_point, duration = self.clip_dur)

        S = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = self.hann_win, hop_length = self.h_len, n_mels = self.nmels, fmin = self.fmin, fmax = self.fmax)
        
        return S

    def audio_to_melspectrogram(self, S):
        
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

    def read_as_melspectrogram(self, path, start_time, end_time):
        S = self.read_audio(path, start_time, end_time)
        mels = self.audio_to_melspectrogram(S)
        return mels

    def convert_wav_to_image(self, df):       
        X = []
        Y = []

        for i, row in tqdm(df.iterrows(), total = len(df.index)):
            file = str(row['sound.files'])
            if os.path.exists(self.audio_path + file):
                    x = self.read_as_melspectrogram(self.audio_path + file, start_time = int(row['start']), end_time = int(row['end']))
                    X.append(x)
                    Y.append(str(row['Annotation']))
            else:
                    print("Cannot find", file)
        
        X = np.array(X)
        Y = np.array(Y)
              
        return X, Y
   
    def save_data_to_pickle(self, X, Y):
        outfile = open(os.path.join(self.save_pickled_data, self.file_name + '_X-pow.pkl'),'wb')
        pickle.dump(X, outfile, protocol=4)
        outfile.close()

        outfile = open(os.path.join(self.save_pickled_data, self.file_name + '_Y-pow.pkl'),'wb')
        pickle.dump(Y, outfile, protocol=4)
        outfile.close()

    def equalize_annotations(self, df):
        ggm_count = len(df[df['Annotation'] == 'GGM'])
        df_other = df[df['Annotation'] != "no"]
        
        df_no = df[df['Annotation'] == 'no'].sample(ggm_count, random_state=1)
        df_final = pd.concat([df_no, df_other])
        
        return df_final   

    def get_data(self):
        # load data
        data = pd.read_csv(self.save_pickled_data + self.file_name + '.csv')

        # ensure we are not processing unnecessary calls
        # training_data = self.equalize_annotations(data)

        # get images
        X_calls, Y_calls = self.convert_wav_to_image(data)

        # save the data
        self.save_data_to_pickle(X_calls, Y_calls)
        
        print('Nailed it')
        
        return X_calls, Y_calls               