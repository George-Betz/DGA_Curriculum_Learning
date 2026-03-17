# evaluation.py

import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Dict

# Ρύθμιση του logger για την καταγραφή των αποτελεσμάτων αξιολόγησης
logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('evaluation.log', mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.propagate = False  # Αποφυγή αναπαραγωγής στο root logger

def evaluate_model(model, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, object]:
    """
    Αξιολογεί το εκπαιδευμένο μοντέλο σε δοκιμαστικά δεδομένα και υπολογίζει μετρήσεις απόδοσης.

    :param model: Το εκπαιδευμένο μοντέλο Keras.
    :param test_data: Τα δεδομένα δοκιμής (κωδικοποιημένα και συμπληρωμένα).
    :param test_labels: Οι ετικέτες δοκιμής.
    :return: Λεξικό με τις μετρήσεις απόδοσης του μοντέλου.
    """
    try:
        # Πρόβλεψη ετικετών για τα δεδομένα δοκιμής
        predictions_prob = model.predict(test_data)
        predictions = (predictions_prob > 0.5).astype("int32")
        logger.info('Η πρόβλεψη των δεδομένων δοκιμής πραγματοποιήθηκε επιτυχώς.')

        # Υπολογισμός μετρήσεων απόδοσης
        model_accuracy = accuracy_score(test_labels, predictions)
        precision_score_val = precision_score(test_labels, predictions, zero_division=0)
        recall_score_val = recall_score(test_labels, predictions, zero_division=0)
        f1_score_val = f1_score(test_labels, predictions, zero_division=0)
        conf_matrix = confusion_matrix(test_labels, predictions)
        class_report = classification_report(test_labels, predictions, zero_division=0)

        logger.info('Υπολογισμός μετρήσεων ολοκληρώθηκε επιτυχώς.')

        return {
            "accuracy": model_accuracy,
            "precision": precision_score_val,
            "recall": recall_score_val,
            "f1_score": f1_score_val,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            "predictions_prob": predictions_prob  
        }
    except Exception as e:
        logger.error(f'Σφάλμα κατά την αξιολόγηση του μοντέλου: {str(e)}')
        raise

def log_evaluation_results(results: Dict[str, object]):
    """
    Καταγράφει τα αποτελέσματα της αξιολόγησης στο αρχείο log 

    :param results: Λεξικό με τις τιμές των μετρήσεων απόδοσης.
    """
    try:
        logger.info("===== Αποτελέσματα Αξιολόγησης =====")
        logger.info(f"Ακρίβεια Μοντέλου (Model Accuracy): {results['accuracy']:.4f}")
        logger.info(f"Προσήλωση (Precision): {results['precision']:.4f}")
        logger.info(f"Ανάκληση (Recall): {results['recall']:.4f}")
        logger.info(f"F1 Score: {results['f1_score']:.4f}")
        logger.info(f"Mήτρα Σύγχυσης (Confusion Matrix):\n{results['confusion_matrix']}")
        logger.info(f"Αναφορά Ταξινόμησης (Classification Report):\n{results['classification_report']}")

        # Άμεση αποθήκευση των αποτελεσμάτων στο αρχείο καταγραφής
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
    except Exception as e:
        logger.error(f'Σφάλμα κατά την καταγραφή των αποτελεσμάτων: {str(e)}')
        raise

def display_evaluation_results(results: Dict[str, object]):
    """
    Εμφάνιση των αποτελεσμάτων αξιολόγησης στην κονσόλα για άμεση ανάλυση.

    :param results: Λεξικό με τις τιμές των μετρήσεων απόδοσης.
    """
    try:
        print("===== Αποτελέσματα Αξιολόγησης =====")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Confusion Matrix:\n{results['confusion_matrix']}")
        print(f"Classification Report:\n{results['classification_report']}")
    except Exception as e:
        logger.error(f'Σφάλμα κατά την εμφάνιση των αποτελεσμάτων: {str(e)}')
        raise

def evaluate_and_log_model(model, test_data: np.ndarray, test_labels: np.ndarray):
    """
    Ολοκληρωμένη διαδικασία αξιολόγησης του μοντέλου με καταγραφή και εμφάνιση των αποτελεσμάτων.

    :param model: Το εκπαιδευμένο μοντέλο Keras.
    :param test_data: Τα δεδομένα δοκιμής (κωδικοποιημένα και συμπληρωμένα).
    :param test_labels: Οι ετικέτες δοκιμής.
    """
    try:
        # Εκτέλεση της αξιολόγησης του μοντέλου
        results = evaluate_model(model, test_data, test_labels)
        
        # Καταγραφή των αποτελεσμάτων
        log_evaluation_results(results)

        # Εμφάνιση των αποτελεσμάτων στην κονσόλα
        display_evaluation_results(results)
        
        logger.info('Η αξιολόγηση του μοντέλου ολοκληρώθηκε επιτυχώς.')
        return results  # Επιστροφή των αποτελεσμάτων για περαιτέρω χρήση
    except Exception as e:
        logger.error(f'Σφάλμα κατά τη διαδικασία αξιολόγησης: {str(e)}')
        raise


if __name__ == "__main__":
    # Παράδειγμα δεδομένων (χρησιμοποιώντας πραγματικά δεδομένα από την προεπεξεργασία)
    from model_training import tokenize_domains
    import pandas as pd

    # Υποθέτουμε ότι έχουμε τα ακόλουθα δεδομένα δοκιμής
    test_domains = np.array(["example.com", "suspiciousdomain.net", "gooddomain.org"])
    test_labels = np.array([0, 1, 0])

    # Φόρτωση του tokenizer που χρησιμοποιήθηκε κατά την εκπαίδευση
    # (Σε πραγματική χρήση, θα πρέπει να σώσουμε και να φορτώσουμε τον tokenizer)
    train_domains = np.array(["normal.com", "badsite123.com", "safesite.net"])
    _, test_data_encoded, tokenizer, max_length = tokenize_domains(train_domains, test_domains)

    # Δημιουργία ενός μοντέλου για το παράδειγμα 
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense
    vocab_size = len(tokenizer.word_index)
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=64, input_length=max_length))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Προσωρινή εκπαίδευση του μοντέλου (για το παράδειγμα)
    train_data_encoded, _, _, _ = tokenize_domains(train_domains, test_domains)
    train_labels = np.array([0, 1, 0])
    model.fit(train_data_encoded, train_labels, epochs=1, verbose=0)

    # Αξιολόγηση του μοντέλου
    evaluate_and_log_model(model, test_data_encoded, test_labels)
