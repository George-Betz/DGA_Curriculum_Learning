# visualization.py

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict

# Ορισμός του logging για καταγραφή σφαλμάτων και πληροφοριών
logging.basicConfig(filename='visualization.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def setup_logger():
    """Δημιουργία και ρύθμιση του logger για καταγραφή σφαλμάτων και πληροφοριών."""
    try:
        logging.info('Ο logger για τις οπτικοποιήσεις αρχικοποιήθηκε επιτυχώς.')
    except Exception as e:
        logging.error(f'Σφάλμα κατά την αρχικοποίηση του logger: {str(e)}')
        raise

def plot_training_results(history: Dict[str, list]):
    """
    Δημιουργεί διαδραστικά γραφήματα για την ακρίβεια και την απώλεια κατά τη διάρκεια της εκπαίδευσης.

    :param history: Το αντικείμενο ιστορικού (history) από την εκπαίδευση του μοντέλου.
    """
    try:
        if 'accuracy' not in history or 'loss' not in history:
            raise ValueError('Το αντικείμενο ιστορικού δεν περιέχει δεδομένα ακρίβειας ή απώλειας.')

        # Γράφημα ακρίβειας
        fig_accuracy = go.Figure()
        fig_accuracy.add_trace(go.Scatter(x=list(range(len(history['accuracy']))), 
                                          y=history['accuracy'], mode='lines+markers', 
                                          name='Training Accuracy', hovertemplate="Epoch %{x}: %{y:.2f}"))
        fig_accuracy.add_trace(go.Scatter(x=list(range(len(history['val_accuracy']))),
                                          y=history['val_accuracy'], mode='lines+markers',
                                          name='Validation Accuracy', hovertemplate="Epoch %{x}: %{y:.2f}"))
        fig_accuracy.update_layout(title='Accuracy During Training',
                                   xaxis_title='Epochs', yaxis_title='Accuracy')
        fig_accuracy.write_html("training_accuracy.html")
        fig_accuracy.show()

        # Γράφημα απώλειας
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=list(range(len(history['loss']))), 
                                      y=history['loss'], mode='lines+markers', 
                                      name='Training Loss', hovertemplate="Epoch %{x}: %{y:.2f}"))
        fig_loss.add_trace(go.Scatter(x=list(range(len(history['val_loss']))), 
                                      y=history['val_loss'], mode='lines+markers', 
                                      name='Validation Loss', hovertemplate="Epoch %{x}: %{y:.2f}"))
        fig_loss.update_layout(title='Loss During Training', 
                               xaxis_title='Epochs', yaxis_title='Loss')
        fig_loss.write_html("training_loss.html")
        fig_loss.show()

        logging.info('Τα διαδραστικά γραφήματα εκπαίδευσης δημιουργήθηκαν επιτυχώς.')
    except Exception as e:
        logging.error(f'Σφάλμα κατά τη δημιουργία των γραφημάτων εκπαίδευσης: {str(e)}')
        raise

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Δημιουργεί διαδραστικό γράφημα για το confusion matrix με tooltips για κάθε τιμή.

    :param conf_matrix: Tο confusion matrix από την αξιολόγηση του μοντέλου.
    :param class_names: Τα ονόματα των κλάσεων που χρησιμοποιούνται για τη διαμόρφωση του πίνακα.
    """
    try:
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix, x=class_names, y=class_names,
            colorscale='Blues', text=conf_matrix, hoverinfo='text',
            hovertemplate="True %{y} Predicted %{x}: %{z}"))

        fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted Label', yaxis_title='True Label')
        fig.write_html("confusion_matrix.html")
        fig.show()
        logging.info('Το διαδραστικό γράφημα του confusion matrix δημιουργήθηκε επιτυχώς.')
    except Exception as e:
        logging.error(f'Σφάλμα κατά τη δημιουργία του γραφήματος confusion matrix: {str(e)}')
        raise

def plot_metric_comparison(curriculum_results, non_curriculum_results):
    """
    Δημιουργεί διαδραστικό γράφημα σύγκρισης μετρήσεων μεταξύ εκπαίδευσης με και χωρίς Curriculum Learning.

    :param curriculum_results: Τα αποτελέσματα με Curriculum Learning.
    :param non_curriculum_results: Τα αποτελέσματα χωρίς Curriculum Learning.
    """
    try:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        curriculum_scores = [curriculum_results['accuracy'], curriculum_results['precision'],
                             curriculum_results['recall'], curriculum_results['f1_score']]
        non_curriculum_scores = [non_curriculum_results['accuracy'], non_curriculum_results['precision'],
                                 non_curriculum_results['recall'], non_curriculum_results['f1_score']]
        
        fig = go.Figure(data=[
            go.Bar(name='Curriculum Learning', x=metrics, y=curriculum_scores, hovertemplate="%{x}: %{y:.2f}"),
            go.Bar(name='No Curriculum Learning', x=metrics, y=non_curriculum_scores, hovertemplate="%{x}: %{y:.2f}")
        ])

        fig.update_layout(title='Comparison of Curriculum Learning vs No Curriculum Learning',
                          barmode='group', xaxis_title='Metrics', yaxis_title='Score')
        fig.write_html("metric_comparison.html")
        fig.show()
        logging.info('Το διαδραστικό γράφημα σύγκρισης των μετρήσεων δημιουργήθηκε επιτυχώς.')
    except Exception as e:
        logging.error(f'Σφάλμα κατά τη δημιουργία του γραφήματος σύγκρισης μετρήσεων: {str(e)}')
        raise

# Ενεργοποίηση του logger
setup_logger()


if __name__ == "__main__":
    # Ψευδοδεδομένα για ιστορικό εκπαίδευσης (για παράδειγμα χρήσης, αλλάζουμε τις τιμές  με πραγματικά δεδομένα)
    history_example = {
        'accuracy': [0.7, 0.8, 0.85, 0.1],
        'val_accuracy': [0.65, 0.75, 0.8, 0.85],
        'loss': [0.5, 0.4, 0.3, 0.9],
        'val_loss': [0.55, 0.45, 0.35, 0.3]
    }

    class DummyHistory:
        def __init__(self, history):
            self.history = history

    dummy_history = DummyHistory(history_example)

    # Δείγματα χρήσης των συναρτήσεων
    plot_training_results(dummy_history.history)

    # Ψευδοπίνακας confusion matrix (πρέπει να αντικατασταθεί με πραγματικά δεδομένα)
    conf_matrix_example = np.array([[675421, 141214], [119737, 63430]])
    plot_confusion_matrix(conf_matrix_example, ['Benign', 'Malicious'])

    # Ψεύτικα δεδομένα για μετρήσεις σύγκρισης
    curriculum_results_example = {'accuracy': 0.9846, 'precision': 0.9339, 'recall': 0.9855, 'f1_score': 0.9590}
    non_curriculum_results_example = {'accuracy': 0.7390, 'precision': 0.3100, 'recall': 0.3463, 'f1_score': 0.3271}
    plot_metric_comparison(curriculum_results_example, non_curriculum_results_example)
