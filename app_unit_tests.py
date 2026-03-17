# app_unit_tests.py

import unittest
from unittest.mock import patch, MagicMock
import logging
import os
import pandas as pd
import numpy as np
import datetime
from data_preprocessing import (
    preprocess_data,
    load_combined_data,
    encode_labels,
    split_easy_difficult,
    stratified_train_test_split,
)
from dga_tranco_data_collector import DGA_TrancoDataCollector
from evaluation import evaluate_model, log_evaluation_results
from model_training import build_model, preprocess_and_train  # Αφαιρέθηκε το 'train_model'
from visualization import (
    plot_training_results,
    plot_confusion_matrix,
    plot_metric_comparison,
)

# Ρύθμιση του logging για λεπτομερή καταγραφή
logger = logging.getLogger("unit_tests")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('test_log.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False  # Αποφυγή αναπαραγωγής των logs σε ανώτερα επίπεδα

class TestDataPreprocessing(unittest.TestCase):
    """
    Κλάση για τη δοκιμή των λειτουργιών προεπεξεργασίας δεδομένων.
    """

    def setUp(self):
        """
        Προετοιμάζει τα δεδομένα δοκιμής πριν από κάθε μέθοδο δοκιμής.
        """
        logger.info("Ρύθμιση των δεδομένων δοκιμής για TestDataPreprocessing.")
        self.mock_data = pd.DataFrame({
            'domain': ['benign.com', 'malicious.com', 'testsite.org', 'safe-site.com'],
            'label': [0, 1, 0, 1]
        })
        self.file_path = 'mock_combined_data.csv'
        self.mock_data.to_csv(self.file_path, index=False)
        logger.info(f"Δημιουργήθηκε το αρχείο mock δεδομένων: {self.file_path}")

    def tearDown(self):
        """
        Καθαρίζει τα δεδομένα δοκιμής μετά από κάθε μέθοδο δοκιμής.
        """
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            logger.info(f"Το αρχείο mock δεδομένων διαγράφηκε: {self.file_path}")

    def test_load_combined_data(self):
        """
        Δοκιμή της συνάρτησης φόρτωσης συνδυασμένων δεδομένων.
        """
        logger.info("Εκτέλεση της δοκιμής test_load_combined_data.")
        data = load_combined_data(self.file_path)
        self.assertFalse(data.empty, "Τα δεδομένα δεν πρέπει να είναι κενά.")
        self.assertEqual(len(data), 4, "Πρέπει να υπάρχουν 4 δείγματα στα δεδομένα.")
        logger.info("Η δοκιμή test_load_combined_data ολοκληρώθηκε επιτυχώς.")

    def test_encode_labels(self):
        """
        Δοκιμή της συνάρτησης κωδικοποίησης ετικετών.
        """
        logger.info("Εκτέλεση της δοκιμής test_encode_labels.")
        encoded_data, label_encoder = encode_labels(self.mock_data)
        self.assertTrue('label' in encoded_data.columns, "Η στήλη 'label' πρέπει να υπάρχει μετά την κωδικοποίηση.")
        self.assertIn(1, encoded_data['label'].values, "Η ετικέτα 1 πρέπει να υπάρχει για κακόβουλους τομείς.")
        logger.info("Η δοκιμή test_encode_labels ολοκληρώθηκε επιτυχώς.")

    def test_split_easy_difficult(self):
        """
        Δοκιμή της συνάρτησης διαχωρισμού εύκολων και δύσκολων παραδειγμάτων.
        """
        logger.info("Εκτέλεση της δοκιμής test_split_easy_difficult.")
        self.mock_data['domain'] = ['ab.com', 'malicioussite.com', 'xyz.org', 'abcd.com']
        easy_data, difficult_data = split_easy_difficult(self.mock_data)
        self.assertGreaterEqual(len(easy_data), 1, "Πρέπει να υπάρχει τουλάχιστον ένα εύκολο παράδειγμα.")
        self.assertGreaterEqual(len(difficult_data), 1, "Πρέπει να υπάρχει τουλάχιστον ένα δύσκολο παράδειγμα.")
        logger.info(f"Εύκολα παραδείγματα: {len(easy_data)}, Δύσκολα παραδείγματα: {len(difficult_data)}")
        logger.info("Η δοκιμή test_split_easy_difficult ολοκληρώθηκε επιτυχώς.")

    def test_stratified_train_test_split(self):
        """
        Δοκιμή της συνάρτησης διαστρωματωμένου διαχωρισμού δεδομένων εκπαίδευσης και δοκιμής.
        """
        logger.info("Εκτέλεση της δοκιμής test_stratified_train_test_split.")
        balanced_data = pd.DataFrame({
            'domain': ['site1.com', 'site2.com', 'site3.com', 'site4.com', 'site5.com', 'site6.com'],
            'label': [0, 1, 0, 1, 0, 1]
        })
        try:
            train_data, test_data = stratified_train_test_split(balanced_data, test_size=0.33)
            self.assertGreaterEqual(len(train_data), 1, "Πρέπει να υπάρχει τουλάχιστον ένα δείγμα εκπαίδευσης.")
            self.assertGreaterEqual(len(test_data), 1, "Πρέπει να υπάρχει τουλάχιστον ένα δείγμα δοκιμής.")
            # Επιπλέον, ελέγχουμε ότι οι ετικέτες είναι ισορροπημένες
            self.assertEqual(train_data['label'].nunique(), 2, "Οι ετικέτες στο train set πρέπει να περιέχουν και τις δύο κλάσεις.")
            self.assertEqual(test_data['label'].nunique(), 2, "Οι ετικέτες στο test set πρέπει να περιέχουν και τις δύο κλάσεις.")
            logger.info("Η δοκιμή test_stratified_train_test_split ολοκληρώθηκε επιτυχώς.")
        except ValueError as e:
            logger.error(f"Ανεπαρκή δεδομένα για διαστρωματωμένο διαχωρισμό: {str(e)}")
            self.fail(f"Η διαστρωματωμένη διαίρεση απέτυχε λόγω ανεπαρκών δεδομένων: {str(e)}")

class TestDGADataCollector(unittest.TestCase):
    """
    Κλάση για τη δοκιμή του συλλέκτη δεδομένων DGA και Tranco.
    """

    @patch('dga_tranco_data_collector.requests.get')
    def test_fetch_dga_data(self, mock_get):
        """
        Δοκιμή της συνάρτησης ανάκτησης DGA δεδομένων από το API.
        """
        logger.info("Εκτέλεση της δοκιμής test_fetch_dga_data.")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "domain1.com\ndomain2.com\n"
        mock_get.return_value = mock_response

        collector = DGA_TrancoDataCollector("2022-01-01", "2022-01-02", "username", "password")
        domains = collector.fetch_dga_data(datetime.date(2022, 1, 1))
        self.assertEqual(len(domains), 2, "Πρέπει να υπάρχουν 2 τομείς στα δεδομένα.")
        self.assertIn('domain1.com', domains, "Το domain1.com πρέπει να υπάρχει στα δεδομένα.")
        self.assertIn('domain2.com', domains, "Το domain2.com πρέπει να υπάρχει στα δεδομένα.")
        logger.info("Η δοκιμή test_fetch_dga_data ολοκληρώθηκε επιτυχώς.")

    def test_filter_benign_domains(self):
        """
        Δοκιμή της συνάρτησης φιλτραρίσματος καλοήθων τομέων από τα DGA δεδομένα.
        """
        logger.info("Εκτέλεση της δοκιμής test_filter_benign_domains.")
        collector = DGA_TrancoDataCollector("2022-01-01", "2022-01-02", "username", "password")
        collector.dga_domains = {'malicious.com'}
        tranco_df = pd.DataFrame({'domain': ['benign.com', 'malicious.com'], 'label': [0, 0]})
        filtered_df = collector.filter_benign_domains(tranco_df)
        self.assertEqual(len(filtered_df), 1, "Πρέπει να υπάρχει ένας καλοήθης τομέας μετά το φιλτράρισμα.")
        self.assertEqual(filtered_df.iloc[0]['domain'], 'benign.com', "Ο καλοήθης τομέας πρέπει να είναι το benign.com.")
        logger.info("Η δοκιμή test_filter_benign_domains ολοκληρώθηκε επιτυχώς.")

class TestEvaluation(unittest.TestCase):
    """
    Κλάση για τη δοκιμή των λειτουργιών αξιολόγησης του μοντέλου.
    """

    def test_evaluate_model(self):
        """
        Δοκιμή της συνάρτησης αξιολόγησης του μοντέλου.
        """
        logger.info("Εκτέλεση της δοκιμής test_evaluate_model.")
        test_data = np.random.rand(10, 50)
        test_labels = np.random.randint(2, size=10)
        model = build_model(input_length=50, vocab_size=100)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(test_data, test_labels, epochs=1, verbose=0)
        results = evaluate_model(model, test_data, test_labels)
        self.assertIn("accuracy", results, "Τα αποτελέσματα αξιολόγησης πρέπει να περιλαμβάνουν την ακρίβεια.")
        logger.info("Η δοκιμή test_evaluate_model ολοκληρώθηκε επιτυχώς.")

    def test_log_evaluation_results(self):
        """
        Δοκιμή ότι τα αποτελέσματα αξιολόγησης καταγράφονται σωστά.
        """
        results = {
            "accuracy": 0.95,
            "precision": 0.9,
            "recall": 0.85,
            "f1_score": 0.87,
            "confusion_matrix": np.array([[8, 2], [1, 9]]),
            "classification_report": "Example Classification Report"
        }

        
        evaluation_logger = logging.getLogger('evaluation')
        evaluation_logger.setLevel(logging.INFO)
        with self.assertLogs(logger='evaluation', level='INFO') as log:
            log_evaluation_results(results)
            log_messages = "\n".join(log.output)
            self.assertIn("Ακρίβεια Μοντέλου (Model Accuracy): 0.95", log_messages, "Το log πρέπει να περιέχει την ακρίβεια του μοντέλου.")
            self.assertIn("Προσήλωση (Precision): 0.9", log_messages, "Το log πρέπει να περιέχει την προσήλωση.")
            self.assertIn("Ανάκληση (Recall): 0.85", log_messages, "Το log πρέπει να περιέχει την ανάκληση.")
            self.assertIn("F1 Score: 0.87", log_messages, "Το log πρέπει να περιέχει το F1 Score.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
