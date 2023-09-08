from custom_model import CustomModel

# create an instance of AudioModelTrainer
trainer = CustomModel(save_pickled_data = "./", INPUT_SHAPE = (128, 87, 3), BS = 32 , EPOCHS = 100, 
                      training_file="training_dataset_split_no_1000", model_name = "5a_(dense1)_layers_training_dataset_split_no_1000_100")

# training the model
model = trainer.train_model()

# get validation performance metrics
trainer.performance(model)

# focal species metrics
trainer.focal_performance(focal_species = ['GGM','SCM'])

