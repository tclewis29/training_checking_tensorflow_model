from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os
import pandas as pd
from datetime import datetime

class CustomModel:
    def __init__(self, save_pickled_data, training_file, EPOCHS, INPUT_SHAPE, BS, model_name):
        self.train_data = training_file
        self.save_pickled_data = save_pickled_data
        self.EPOCHS = EPOCHS
        self.INPUT_SHAPE = INPUT_SHAPE
        self.BS = BS
        self.model_name = model_name

    def save_data_to_pickle(self, X, Y, file_name):
        outfile = open(os.path.join(self.save_pickled_data, file_name + '_test_X-pow.pkl'),'wb')
        pickle.dump(X, outfile, protocol=4)
        outfile.close()

        outfile = open(os.path.join(self.save_pickled_data, file_name + '_test_Y-pow.pkl'),'wb')
        pickle.dump(Y, outfile, protocol=4)
        outfile.close()

    def load_pickled_data(self, file_name):
        '''
        Load all of the spectrograms from a pickle file
        '''
        X_data_name = file_name + '_X-pow.pkl'
        Y_data_name = file_name + '_Y-pow.pkl'
             
        infile = open(os.path.join(self.save_pickled_data, X_data_name),'rb')
        X = pickle.load(infile)
        infile.close()
     
        infile = open(os.path.join(self.save_pickled_data, Y_data_name),'rb')
        Y = pickle.load(infile)
        infile.close()

        return X, Y
    
    def categorical_to_numerical(self, labels):
        unique_labels = list(set(labels))  # getting the unique labels
        num_labels = []

        # creating a dictionary to map labels to numbers
        self.label_dict = {label: num for num, label in enumerate(unique_labels)}
        
        # creating a dictionary to map numbers to labels
        self.num_dict = {v: k for k, v in self.label_dict.items()}

        # printing the relationships
        print(f"Label to Number mapping: {self.label_dict}")
        print(f"Number to label mapping: {self.num_dict}")

        # mapping the labels to their numerical values
        for k in labels:
            num_labels.append(self.label_dict[k])

        # Save mapping to CSV
        self.save_label_mapping_to_csv()

        return num_labels

    def save_label_mapping_to_csv(self):
        label_mapping_df = pd.DataFrame(list(self.label_dict.items()), columns=['Label', 'Number'])
        label_mapping_df.to_csv('class_labels.csv', index=False)

    def numerical_to_categorical(self, num_labels):
        # using num_dict to convert numbers back to labels
        cat_labels = [self.num_dict[label] for label in num_labels]

        print(f"Number to label mapping: {self.num_dict}")
        return cat_labels

    def train_model(self, X=None, Y=None):
        if X is None or Y is None:
            self.trainX, self.trainY = self.load_pickled_data(self.train_data)
        else:
            self.trainX = X
            self.trainY = Y

        unique, counts = np.unique(self.trainY, return_counts=True)
        original_distribution = dict(zip(unique, counts))
        print('Original Data distribution:',original_distribution)

        X_train, self.X_test, y_train, self.y_test = train_test_split(self.trainX, self.trainY, test_size=0.2, random_state=42)

        self.save_data_to_pickle(self.X_test, self.y_test, self.train_data)
        
        # Calculate the number of unique classes
        num_classes = len(np.unique(y_train))

        # Hot label training data
        Y_cat = self.categorical_to_numerical(y_train) 
        Y_one_hot = tf.one_hot(Y_cat, depth=num_classes)
        
        # begin model creation:
        model=Sequential()
        
        #covolution layer 1
        model.add(Conv2D(32,(3,3),activation='relu', padding='same', input_shape=self.INPUT_SHAPE))
        #pooling layer
        model.add(MaxPooling2D(2,2))
        model.add(BatchNormalization())
        
        #covolution layer 2
        model.add(Conv2D(32,(3,3),activation='relu', padding='same'))
        #pooling layer
        model.add(MaxPooling2D(2,2))
        model.add(BatchNormalization())
        
        #covolution layer 3
        model.add(Conv2D(64,(3,3),activation='relu', padding='same'))
        #pooling layer
        model.add(MaxPooling2D(2,2))
        model.add(BatchNormalization())
        
        #covolution layer 4
        model.add(Conv2D(64,(3,3),activation='relu', padding='same'))
        #pooling layer
        model.add(MaxPooling2D(2,2))
        model.add(BatchNormalization())

        #covolution layer 5
        model.add(Conv2D(64,(3,3),activation='relu', padding='same'))
        #pooling layer
        model.add(MaxPooling2D(2,2))
        model.add(BatchNormalization())
            
        #i/p layer
        model.add(Flatten())

        # first dense layer
        model.add(Dense(128, activation='relu'))
       
        #o/p layer
        model.add(Dense(num_classes,activation='softmax'))

        # complile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5),metrics=['accuracy'])

        # generate a model summary
        model.summary()

        # define the early stopping, when val_loss curve flattens the training stops
        early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)

        # fit the model
        history=model.fit(X_train,Y_one_hot,validation_split=0.2,epochs=self.EPOCHS,callbacks=[early_stop],shuffle=True, batch_size = self.BS)
        
        #loss graph
        plt.plot(history.history['loss'],label='train loss')
        plt.plot(history.history['val_loss'],label='val loss')
        plt.legend()
        plt.savefig(self.model_name + 'loss-graph.png')
        plt.show()
        # accuracies
        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='val acc')
        plt.legend()
        
        plt.savefig(self.model_name + 'acc-graph.png')
        plt.show()

        # save the model
        model.save("./" + self.model_name + "_model.h5")
        print('Model saved as', self.model_name, '_model.h5')

        return model

    def performance(self, model):
        print('Evaluating performance...')
        unique, counts = np.unique(self.y_test, return_counts=True)
        original_distribution = dict(zip(unique, counts))
        print('Test data distribution:',original_distribution)

        y_pred=model.predict(self.X_test)
        self.y_pred=np.argmax(y_pred,axis=1)

        y_pred_cat = self.numerical_to_categorical(self.y_pred)
      
        # classification report
        print(classification_report(y_pred_cat,self.y_test))
        class_report_dict = classification_report(y_pred_cat,self.y_test, output_dict=True)
        
        # Convert to DataFrame
        class_report_df = pd.DataFrame(class_report_dict).transpose()
        current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        class_report_df.to_csv(f'validation_classification_report_{current_timestamp}.csv', index=True)
        
        # create, save and display the confusion matrix
        print(np.shape(self.y_test))
        print(np.shape(y_pred_cat))
        ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred_cat, normalize = 'true')
        plt.show()
        plt.savefig(self.model_name + 'validation_confusion_matrix.png')

    def focal_performance(self, focal_species):
            
            # Make sure focal_species is a list
            if isinstance(focal_species, str):
                focal_species = [focal_species]

            # If self.y_pred is a single integer, convert it to a list
            if np.isscalar(self.y_pred):
                Y_pred = [self.y_pred]
            else:
                Y_pred = self.y_pred.tolist()
            
            # Dynamic grouping of all classes other than those in focal_species into a "no" category
            Y_pred = ['no' if self.num_dict[id] not in focal_species else self.num_dict[id] for id in Y_pred]
            Y_true = ['no' if y not in focal_species else y for y in self.y_test.tolist()]

            # Dynamic labeling
            labels = focal_species + ['no']

            # reference data class summaries
            unique, counts = np.unique(Y_true, return_counts=True)
            ref_distribution = dict(zip(unique, counts))
            print('True reduced data distribution:',ref_distribution) 

            # Predicted data class summaries
            unique, counts = np.unique(Y_pred, return_counts=True)
            pred_distribution = dict(zip(unique, counts))
            print('Predicted reduced data distribution:',pred_distribution) 
            
            current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            cm = confusion_matrix(Y_true, Y_pred, labels=labels)
            class_report_dict = classification_report(Y_true, Y_pred, output_dict=True)
            class_report_df = pd.DataFrame(class_report_dict).transpose()
            class_report_df.to_csv(f'focal_species_report_{current_timestamp}.csv', index=True)
            ConfusionMatrixDisplay.from_predictions(Y_true, Y_pred, normalize='true')
            plt.savefig(self.model_name + 'focal_species_confusion_matrix.png')

            confusion_df = pd.DataFrame(cm, index=labels, columns=labels)
            
            # Dynamic calculation and printing of performance metrics
            for species in focal_species:
                tp = confusion_df.loc[species, species] if confusion_df.loc[species, species] > 0 else 0
                tn = confusion_df.drop(species).drop(species, axis=1).values.sum() if confusion_df.drop(species).drop(species, axis=1).values.sum() > 0 else 0
                fp = confusion_df.drop(species).loc[:, species].sum() if confusion_df.drop(species).loc[:, species].sum() > 0 else 0
                fn = confusion_df.loc[species].drop(species).sum() if confusion_df.loc[species].drop(species).sum() > 0 else 0

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                # Round to 4 significant figures
                precision = round(precision, 3)
                recall = round(recall, 3)

                print(f"\nSummary for {species}")
                print(f'True Positives: {tp}')
                print(f'True Negatives: {tn}')
                print(f'False Positives: {fp}')
                print(f'False Negatives: {fn}')
                print(f'For {species} - Precision: {precision}, Recall: {recall}')