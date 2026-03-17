# data_preprocessing.py

import pandas as pd
import os
import logging
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# Διαμόρφωση του logging με υποστήριξη για UTF-8 για παρακολούθηση σφαλμάτων και πληροφοριών
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

def setup_logger():
    """
    Αρχικοποιεί τον logger για την αποφυγή διπλών καταχωρήσεων και την παρακολούθηση σφαλμάτων και πληροφοριών.
    """
    if logger.hasHandlers():
        logger.handlers.clear()  # Αφαίρεση παλαιών handlers για αποφυγή διπλοεγγραφών

    # Δημιουργία και ρύθμιση του file handler με UTF-8 για την καταγραφή σε αρχείο
    file_handler = logging.FileHandler("data_preprocessing.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.propagate = False  # Αποφυγή διπλοεγγραφών σε άλλους loggers
    logger.debug('Ο logger για την προεπεξεργασία δεδομένων αρχικοποιήθηκε επιτυχώς.')

# Ενεργοποίηση του logger
setup_logger()

def load_combined_data(file_path):
    """
    Φορτώνει το συνδυασμένο αρχείο CSV που περιέχει τα καλοήθη και κακόβουλα domains,
    διασφαλίζοντας την ακεραιότητα των δεδομένων και βελτιστοποιώντας τη μνήμη.

    :param file_path: Η διαδρομή του αρχείου δεδομένων.
    :return: DataFrame που περιέχει τα καλοήθη και κακόβουλα δεδομένα.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Το αρχείο δεδομένων δεν βρέθηκε: {file_path}")

        # Φόρτωση των δεδομένων με καθορισμό τύπων δεδομένων για βελτιστοποίηση μνήμης
        data = pd.read_csv(file_path, dtype={'domain': 'string', 'label': 'int8'})

        if data.empty:
            raise ValueError("Το αρχείο εισόδου είναι κενό.")

        # Αφαίρεση πιθανών διπλότυπων με βάση τη στήλη 'domain'
        initial_count = len(data)
        data.drop_duplicates(subset='domain', inplace=True)
        final_count = len(data)
        logger.info(f'Επιτυχής φόρτωση δεδομένων από {file_path}, συνολικά δείγματα: {final_count} (αφαιρέθηκαν {initial_count - final_count} διπλότυπα)')
        return data

    except FileNotFoundError as fnf_error:
        logger.error(f'Σφάλμα αρχείου: {str(fnf_error)}')
        raise
    except pd.errors.EmptyDataError:
        logger.error("Το αρχείο δεδομένων είναι κενό.")
        raise
    except Exception as e:
        logger.error(f'Σφάλμα κατά τη φόρτωση των δεδομένων: {str(e)}')
        raise

def validate_data(df, expected_columns=['domain', 'label']):
    """
    Εξασφαλίζει ότι οι απαιτούμενες στήλες υπάρχουν στο DataFrame και ελέγχει για τυχόν κενές τιμές.

    :param df: DataFrame προς επικύρωση.
    :param expected_columns: Λίστα με τις απαιτούμενες στήλες.
    :raise KeyError: Αν κάποια στήλη λείπει.
    """
    try:
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f'Λείπουν οι στήλες: {missing_columns}')

        if df.isnull().any().any():
            logger.warning("Τα δεδομένα περιέχουν κενές τιμές. Εξετάστε τη διαχείριση των ελλείψεων.")

    except KeyError as e:
        logger.error(f'Σφάλμα κατά την επιβεβαίωση δεδομένων: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'Απρόσμενο σφάλμα κατά την επιβεβαίωση των δεδομένων: {str(e)}')
        raise

def encode_labels(data):
    """
    Κωδικοποιεί τις ετικέτες σε δυαδική μορφή (0: καλοήθη, 1: κακόβουλα).

    :param data: DataFrame προς κωδικοποίηση.
    :return: DataFrame με τις κωδικοποιημένες ετικέτες και το αντικείμενο LabelEncoder.
    """
    try:
        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])
        logger.debug('Οι ετικέτες κωδικοποιήθηκαν επιτυχώς.')
        return data, label_encoder
    except KeyError as e:
        logger.error(f'Σφάλμα: Η στήλη ετικέτας δεν βρέθηκε για κωδικοποίηση: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'Σφάλμα κατά την κωδικοποίηση των ετικετών: {str(e)}')
        raise

def split_easy_difficult(data):
    """
    Χωρίζει τα δεδομένα σε εύκολα και δύσκολα παραδείγματα με βάση τα χαρακτηριστικά του domain.

    :param data: DataFrame προς διαχωρισμό.
    :return: DataFrames για τα εύκολα και τα δύσκολα παραδείγματα.
    """
    try:
        # Υπολογισμός μήκους του domain και αριθμού μοναδικών χαρακτήρων για κάθε domain
        domain_lengths = data['domain'].str.len()
        unique_chars = data['domain'].apply(lambda x: len(set(x)))

        # Δημιουργία μασκών για εύκολα και δύσκολα παραδείγματα
        easy_mask = (domain_lengths < 11) & (unique_chars > 4)
        difficult_mask = ~easy_mask

        # Διαχωρισμός των δεδομένων χρησιμοποιώντας τις μάσκες
        easy_data = data[easy_mask]
        difficult_data = data[difficult_mask]

        logger.debug(f'Εύκολα παραδείγματα: {len(easy_data)}, Δύσκολα παραδείγματα: {len(difficult_data)}')
        return easy_data, difficult_data
    except KeyError as e:
        logger.error(f'Σφάλμα κλειδιού: Η στήλη domain δεν βρέθηκε: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'Σφάλμα κατά το διαχωρισμό των δεδομένων: {str(e)}')
        raise

def stratified_train_test_split(data, test_size=0.2):
    """
    Διαχωρίζει τα δεδομένα σε σετ εκπαίδευσης και δοκιμής με διαστρωματωμένο τρόπο,
    διατηρώντας την ισορροπία των κλάσεων. Σε περίπτωση που τα δεδομένα είναι ανεπαρκή
    για διαστρωματωμένο διαχωρισμό, χρησιμοποιείται ένας απλός διαχωρισμός.

    :param data: Το DataFrame με τα δεδομένα προς διαχωρισμό.
    :param test_size: Το ποσοστό των δεδομένων που θα διατηρηθεί για δοκιμή.
    :return: Τα σετ εκπαίδευσης και δοκιμής.
    """
    try:
        # Έλεγχος αν υπάρχουν αρκετά δεδομένα για διαστρωματωμένο διαχωρισμό
        label_counts = data['label'].value_counts()
        if any(label_counts < 2) or len(data) * test_size < 2:
            # Εάν τα δεδομένα δεν επαρκούν για διαστρωματωμένο διαχωρισμό, χρησιμοποιείται απλός διαχωρισμός
            logging.warning("Μη επαρκή δεδομένα για διαστρωματωμένο διαχωρισμό. Χρησιμοποιείται απλός διαχωρισμός.")
            return train_test_split(data, test_size=test_size, random_state=42)

        # Εφαρμογή διαστρωματωμένου διαχωρισμού
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        for train_idx, test_idx in splitter.split(data, data['label']):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

        logging.debug("Τα δεδομένα χωρίστηκαν επιτυχώς σε σετ εκπαίδευσης και δοκιμής με διαστρωματωμένο διαχωρισμό.")
        return train_data, test_data

    except ValueError as e:
        logging.error(f"Σφάλμα κατά τον διαχωρισμό δεδομένων: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Απρόσμενο σφάλμα κατά τον διαχωρισμό δεδομένων: {str(e)}")
        raise

def preprocess_data(file_path):
    """
    Πλήρης διαδικασία προεπεξεργασίας δεδομένων: φόρτωση, επικύρωση, κωδικοποίηση και διαχωρισμός
    σε εύκολα/δύσκολα και εκπαίδευσης/δοκιμής.

    :param file_path: Διαδρομή προς το συνδυασμένο αρχείο δεδομένων.
    :return: Εύκολα και δύσκολα δεδομένα σε σετ εκπαίδευσης και δοκιμής.
    """
    try:
        data = load_combined_data(file_path)
        validate_data(data)
        data, label_encoder = encode_labels(data)

        easy_data, difficult_data = split_easy_difficult(data)

        easy_train, easy_test = stratified_train_test_split(easy_data)
        difficult_train, difficult_test = stratified_train_test_split(difficult_data)

        logger.debug('Η πλήρης προεπεξεργασία των δεδομένων ολοκληρώθηκε επιτυχώς.')
        return easy_train, easy_test, difficult_train, difficult_test

    except NotFittedError as e:
        logger.error(f'Σφάλμα κατά την κωδικοποίηση: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'Σφάλμα κατά την πλήρη προεπεξεργασία των δεδομένων: {str(e)}')
        raise


if __name__ == "__main__":
    combined_data_path = os.path.join(os.path.dirname(__file__), "combined_data", "combined_dga_tranco_data.csv")
    easy_train, easy_test, difficult_train, difficult_test = preprocess_data(combined_data_path)
