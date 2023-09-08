from predict import AudioPredictor

# create an instance of AudioPredictor
predictor = AudioPredictor(test = "Yes", audio_path = "E:/2020 PhD/Abundance/", class_labels = "class_labels.csv", downsample_rate = 22050, INPUT_SHAPE = (128, 87, 3),
                           hann_win = 1024, h_len = 256, nmels = 128, fmin = 500, fmax = 9000)

# If test is set to yes we need to define the test variables
predictor.get_test_info(sample_size= 10, test_csv = "./test_soundfiles.csv")

# predict
results = predictor.predict_and_save(model_name="5a_(dense1)_layers_training_dataset_split_no_1000_100_model.h5")