import pandas as pd
import os
from datetime import datetime

class AudioChecker:
    def __init__(self, input_data, txt_files_dir):
        self.txt_files_dir = txt_files_dir  # path to folder with reference selection tables
        self.results_list = []
        
        # Check if input_data is a DataFrame or a CSV file path
        if isinstance(input_data, pd.DataFrame):
            self.csv_df = input_data
        elif isinstance(input_data, str):
            self.csv_df = pd.read_csv(input_data)
        else:
            raise ValueError("input_data must be either a pandas DataFrame or a path to a CSV file.")

        # Counting the number of unique sound files
        unique_sound_files_count = self.csv_df['sound_files'].nunique()

        print(f"Number of unique sound files: {unique_sound_files_count}")

    def analyze(self):
        txt_files_list = [f for f in os.listdir(self.txt_files_dir) if f.endswith('.txt')]
        unique_sound_files = self.csv_df['sound_files'].unique()
        
        for sound_file in unique_sound_files:
            for txt_file in txt_files_list:
                txt_file_first_part = txt_file.split('.')[0]
                
                if sound_file in txt_file_first_part + '.WAV':
                    txt_df = pd.read_csv(os.path.join(self.txt_files_dir, txt_file), delimiter='\t')
                    csv_filtered = self.csv_df[self.csv_df['sound_files'] == sound_file]
                    
                    for idx, csv_row in csv_filtered.iterrows():
                        csv_start = csv_row['start']
                        csv_end = csv_row['end']
                        csv_label = csv_row['label']
                        
                        overlapping_rows = txt_df[(txt_df['Begin Time (s)'] <= csv_end) & (txt_df['End Time (s)'] >= csv_start)]
                        
                        txt_label = None
                        match_scm = match_ggm = False
                        
                        if not overlapping_rows.empty:
                            for _, overlap_row in overlapping_rows.iterrows():
                                txt_label = overlap_row['Annotation']
                                match_scm = (txt_label == 'SCM' and csv_label == 'SCM')
                                match_ggm = (txt_label == 'GGM' and csv_label == 'GGM')
                        
                        else:
                            if csv_label == 'no':
                                match_scm = True
                            if csv_label == 'no':
                                match_ggm = True
                        
                        new_row = {
                            'sound_file': sound_file,
                            'csv_start': csv_start,
                            'csv_end': csv_end,
                            'csv_label': csv_label,
                            'txt_label': txt_label,
                            'match_scm': match_scm,
                            'match_ggm': match_ggm,
                        }
                        self.results_list.append(new_row)

    def calculate_focal_species_performance(self, focal_species):
        results_df = pd.DataFrame(self.results_list)
        results_df.to_csv('cross_reference_results.csv', index=False)

        # Initialize counters
        metrics = {species: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for species in focal_species}

        # Loop through the results_df to compute metrics
        for _, row in results_df.iterrows():
            predicted_label = row['csv_label']
            true_label = row['txt_label']

            for species in focal_species:
                if predicted_label == species:
                    if true_label == species:
                        metrics[species]['tp'] += 1
                    else:
                        metrics[species]['fp'] += 1
                else:
                    if true_label == species:
                        metrics[species]['fn'] += 1
                    else:
                        metrics[species]['tn'] += 1

        # Calculate precision and recall
        performance_metrics = {}
        for species, counts in metrics.items():
            tp, tn, fp, fn = counts['tp'], counts['tn'], counts['fp'], counts['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Round to 4 significant figures
            precision = round(precision, 3)
            recall = round(recall, 3)

            performance_metrics[species] = {'Precision': precision, 'Recall': recall}

            print(f"\nRaw metrics for {species}")
            print(f'True Positives: {tp}')
            print(f'True Negatives: {tn}')
            print(f'False Positives: {fp}')
            print(f'False Negatives: {fn}')
            print(f'For {species} - Precision: {precision}, Recall: {recall}')

        # Save performance metrics to DataFrame
        self.performance_df = pd.DataFrame(performance_metrics).transpose()
        self.meta_df = pd.DataFrame(metrics).transpose()

        self.performance_df.to_csv('test_focal_species_performance_metrics.csv', index=True)
        self.meta_df.to_csv('test_focal_species_meta_metrics.csv', index=True)

    def save_metrics(self):
        # Get the current timestamp
        current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
       
        # Add n_soundfiles column before saving
        unique_sound_files_count = self.csv_df['sound_files'].nunique()
        self.performance_df = self.performance_df.assign(n_soundfiles=unique_sound_files_count)
        self.meta_df = self.meta_df.assign(n_soundfiles=unique_sound_files_count)

        # Save the DataFrames to CSV files
        self.performance_df.to_csv(f'test_performance_metrics_{current_timestamp}.csv', index=False)
        self.meta_df.to_csv(f'test_raw_metrics_{current_timestamp}.csv', index=False)

    def run(self, focal_species = None):
        self.analyze()
        if focal_species:
            self.calculate_focal_species_performance(focal_species)
        self.save_metrics()
        print('Done')