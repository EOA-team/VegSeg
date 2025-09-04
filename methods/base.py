import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score, confusion_matrix
import PIL

pd.options.display.max_columns = None


class BenchmarkMethod:
    @staticmethod
    def calculate_metrics(preds: np.array, y: np.array):
        classes_labels = {'plants': 0, 'soil': 1}

        score_names = ['f1', 'precision', 'recall', 'IoU']
        columns = [name + ' - class: ' + class_label for name in score_names for class_label in classes_labels.keys()]
        columns.append('accuracy: ')

        # Initialize an empty list to accumulate rows
        statistics = []

        # Iterate sample-wise through the data and calculate metrics
        for pred_i, y_i in zip(preds, y):
            scores = {}

            for class_label, i in classes_labels.items():
                # Calculate metrics
                scores['f1 - class: ' + class_label] = f1_score(y_i, pred_i, average='binary', pos_label=i, zero_division=0)
                scores['precision - class: ' + class_label] = precision_score(y_i, pred_i, average='binary', pos_label=i, zero_division=0)
                scores['recall - class: ' + class_label] = recall_score(y_i, pred_i, average='binary', pos_label=i, zero_division=0)
                scores['IoU - class: ' + class_label] = jaccard_score(y_i, pred_i, average='binary', pos_label=i, zero_division=0)

            scores['accuracy: '] = accuracy_score(y_i, pred_i)

            # Append scores as a Series to the list
            statistics.append(pd.Series(scores))

        # Concatenate all Series into a DataFrame
        statistics_df = pd.concat(statistics, axis=1).T

        # Calculate mean of metrics across samples
        mean_metrics = statistics_df.mean(axis=0).to_dict()

        # Convert mean_metrics to have the same format as before
        return {key: mean_metrics[key] for key in columns}

    @staticmethod
    def read_image(imagepath, rgb: bool, metadata: bool = False):
        # read image in a standardized manner
        image = PIL.Image.open(imagepath)

        if metadata:
            metadata = image.text

        if rgb:
            pass
        else:
            image = image.convert('L')

        image = np.array(image)

        # normalize to [0,1]
        if image.max() > 1:  # Assume uint8 format
            image = image / 255.0

        if not rgb:
            # Encode the greyscale masks
            image = np.where(image > 0.5, np.ones_like(image), np.zeros_like(image))

        if metadata:
            return image, metadata

        return image

