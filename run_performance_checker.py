from performance_checker import AudioChecker

# create an instance of AudioChecker
checker = AudioChecker(txt_files_dir = "C:/Users/bi1tl/Documents/Manually_labelled", input_data = "predictions.csv")

# measure performance
checker.run(focal_species=['GGM','SCM'])