from preprocessing import AudioProcessor

# create an instance of AudioProcessor
processor = AudioProcessor(audio_path = "E:/2020 PhD/Abundance/", save_pickled_data = "./", clip_dur = 1, file_name = "training_split_07-07-23_1500", # change this to the current date
                           downsample_rate = 22050, hann_win = 1024, h_len = 256, nmels = 128, fmin = 500, fmax = 9000)

# use the methods of the AudioProcessor class
X_Calls, Y_Calls = processor.get_data(training_files = "./training_dataset_split_1500.csv") # the csv has to have columns: sound.files, start, end, Annotation
