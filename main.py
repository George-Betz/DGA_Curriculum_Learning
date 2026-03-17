# main.py

# credentials : george_betzelos  unfoldoverlaidtuesdaynegligentcarrotguacamole

import csv
import sys
import logging
import os
import pandas as pd
import tensorflow as tf
from collections import Counter
from data_preprocessing import preprocess_data
from model_training import preprocess_and_train
from evaluation import evaluate_and_log_model
from visualization import plot_training_results, plot_confusion_matrix, plot_metric_comparison
from dga_tranco_data_collector import DGA_TrancoDataCollector
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Ορισμός μέγιστου μεγέθους πεδίου CSV για αποφυγή σφαλμάτων σε μεγάλα πεδία
try:
    csv.field_size_limit(int(1e9))
except OverflowError:
    csv.field_size_limit(sys.maxsize)

def setup_logger():
    """
    Ρυθμίζει τον κύριο logger για την παρακολούθηση σφαλμάτων και πληροφοριών.
    Προσθέτει επίσης console handler για αναλυτική απεικόνιση στην κονσόλα.
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()  # Καθαρισμός των handlers για αποφυγή διπλών καταχωρήσεων

    # Διαχειριστής αρχείων για logging
    file_handler = logging.FileHandler('main.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler για εκτύπωση στην κονσόλα
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Ρύθμιση επιπέδου καταγραφής
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)
    logger.info('Ο logger για την κύρια ροή εργασίας αρχικοποιήθηκε επιτυχώς.\n')

def check_device():
    """
    Ελέγχει αν υπάρχει υποστήριξη GPU ή αν η εκπαίδευση θα εκτελεστεί σε CPU.
    """
    logger = logging.getLogger()
    if tf.config.list_physical_devices('GPU'):
        device = "GPU"
        logger.info("Η εκπαίδευση θα εκτελεστεί στην GPU.\n")
    else:
        device = "CPU"
        tf.config.optimizer.set_jit(True)  # Ενεργοποίηση XLA για βελτιστοποίηση CPU
        logger.info("Η εκπαίδευση θα εκτελεστεί στην CPU (Intel-optimized TensorFlow με oneDNN).\n")
    return device

def check_combined_csv():
    """
    Ελέγχει αν υπάρχει το συνδυασμένο αρχείο CSV στον φάκελο 'combined_data'.
    """
    combined_data_path = "combined_data/combined_dga_tranco_data.csv"
    csv_exists = os.path.exists(combined_data_path)
    logger = logging.getLogger()
    if csv_exists:
        logger.info(f"Βρέθηκε το αρχείο δεδομένων: {combined_data_path}\n")
    else:
        logger.info("Δεν βρέθηκε το αρχείο δεδομένων. Θα χρειαστεί να συλλέξετε τα δεδομένα.\n")
    return csv_exists, combined_data_path

def validate_and_analyze_csv(file_path):
    """
    Ελέγχει τη δομή του CSV για την εκπλήρωση των απαιτήσεων της εφαρμογής.
    
    """
    logger = logging.getLogger()
    try:
        logger.info(f"Επικύρωση και ανάλυση του αρχείου CSV: {file_path}\n")
        if not os.path.exists(file_path):
            logger.error(f"Το αρχείο CSV '{file_path}' δεν βρέθηκε.\n")
            return False

        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            # Έλεγχος απαιτούμενων στηλών
            
            required_columns = {'domain', 'label'}
            if set(header) != required_columns:
                logger.error("Το CSV δεν περιέχει τις απαιτούμενες στήλες 'domain' και 'label'.\n")
                return False

            logger.info("Επιτυχία: Οι απαιτούμενες στήλες υπάρχουν.\n")

            # Στατιστική ανάλυση του αρχείου CSV
            row_count = 0
            unique_domains = set()
            domain_lengths = []
            label_counts = Counter()
            errors_found = False
            missing_values = 0

            for row in reader:
                row_count += 1
                if len(row) != 2:
                    logger.error(f"Η γραμμή {row_count} έχει λανθασμένο αριθμό στηλών.\n")
                    errors_found = True
                    continue

                domain, label = row
                if not domain or not label:
                    logger.warning(f"Η γραμμή {row_count} περιέχει κενή τιμή.\n")
                    missing_values += 1
                    errors_found = True

                unique_domains.add(domain)
                domain_lengths.append(len(domain))

                if label not in {'0', '1'}:
                    logger.error(f"Η γραμμή {row_count}: Μη έγκυρη ετικέτα '{label}' (αναμένεται 0 ή 1).\n")
                    errors_found = True
                else:
                    label_counts[label] += 1

            avg_domain_length = sum(domain_lengths) / len(domain_lengths) if domain_lengths else 0
            logger.info(f"===== Στατιστικά Αρχείου CSV =====")
            logger.info(f"Συνολικές γραμμές: {row_count}")
            logger.info(f"Μοναδικά domains: {len(unique_domains)}")
            logger.info(f"Μέσο μήκος domain: {avg_domain_length:.2f}")
            logger.info(f"Κατανομή ετικετών: {label_counts}")
            logger.info(f"Κενές τιμές: {missing_values}")
            logger.info(f"Σφάλματα κατά την ανάλυση: {'Ναι' if errors_found else 'Όχι'}\n")

            if errors_found:
                logger.error("Το CSV περιέχει σφάλματα. Ελέγξτε τα παραπάνω ζητήματα.\n")
                return False
            else:
                logger.info("Το CSV πέρασε όλους τους ελέγχους και είναι έγκυρο.\n")
                return True

    except csv.Error as e:
        logger.error(f"Κρίσιμο σφάλμα κατά την ανάλυση του CSV: {e}\n")
        return False
    except Exception as e:
        logger.error(f"Απροσδόκητο σφάλμα: {e}\n")
        return False

def collect_data(username, password):
    """
    Συλλέγει δεδομένα DGA και Tranco και τα αποθηκεύει για περαιτέρω επεξεργασία.
    Παρέχει αναλυτική ενημέρωση κατά τη διαδικασία.
    """
    logger = logging.getLogger()
    try:
        logger.info('Ξεκινά η συλλογή δεδομένων DGA και Tranco...\n')
        collector = DGA_TrancoDataCollector(username=username, password=password)
        collector.run()
        logger.info('Η συλλογή δεδομένων ολοκληρώθηκε επιτυχώς.\n')
    except Exception as e:
        logger.error(f"Σφάλμα κατά τη συλλογή δεδομένων: {str(e)}\n")
        raise

def run_pipeline(data_path, use_curriculum=False):
    """
    Εκτελεί την πλήρη ροή εργασιών για προεπεξεργασία, εκπαίδευση, αξιολόγηση και οπτικοποίηση δεδομένων.
    Παρέχει αναλυτική ενημέρωση κατά τη διαδικασία.
    """
    logger = logging.getLogger()
    try:
        # Έλεγχος αν υπάρχει αποθηκευμένο μοντέλο
        saved_model_path = 'saved_models/dga_detection_model.keras'
        if os.path.exists(saved_model_path):
            user_input = input("Βρέθηκε αποθηκευμένο μοντέλο. Θέλετε να το φορτώσετε; (yes/no): ").strip().lower()
            if user_input == 'yes':
                logger.info('Φόρτωση του αποθηκευμένου μοντέλου...\n')
                model = load_model(saved_model_path)
                logger.info('Το μοντέλο φορτώθηκε επιτυχώς.\n')

                # Φόρτωση του tokenizer και του max_length
                with open('saved_models/tokenizer.pickle', 'rb') as handle:
                    tokenizer = pickle.load(handle)
                with open('saved_models/max_length.pickle', 'rb') as handle:
                    max_length = pickle.load(handle)

                # Φόρτωση των δεδομένων δοκιμής
                logger.info('Ξεκινά η προεπεξεργασία των δεδομένων για δοκιμή...\n')
                _, _, _, difficult_test = preprocess_data(data_path)
                test_data = difficult_test['domain']
                test_labels = difficult_test['label']

                # Κωδικοποίηση και συμπλήρωση των δεδομένων δοκιμής
                test_sequences = tokenizer.texts_to_sequences(test_data.values)
                test_data_encoded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

                logger.info('Ξεκινά η αξιολόγηση του μοντέλου...\n')
                results = evaluate_and_log_model(model, test_data_encoded, test_labels.values)
                logger.info('Η αξιολόγηση του μοντέλου ολοκληρώθηκε.\n')

                logger.info('Δημιουργία γραφήματος Confusion Matrix...\n')
                plot_confusion_matrix(results['confusion_matrix'], ['Benign', 'Malicious'])
                logger.info('Το γράφημα Confusion Matrix δημιουργήθηκε επιτυχώς.\n')

                # Ερώτηση στον χρήστη αν θέλει να προχωρήσει σε οπτικοποίηση
                user_input = input("Θέλετε να προχωρήσετε στην οπτικοποίηση των αποτελεσμάτων; (yes/no): ").strip().lower()
                if user_input == 'yes':
                    logger.info('Δημιουργία γραφημάτων για τα αποτελέσματα εκπαίδευσης...\n')
                    # Υποθέτοντας ότι έχουμε αποθηκεύσει το ιστορικό εκπαίδευσης
                    with open('saved_models/training_history.pickle', 'rb') as handle:
                        history_dict = pickle.load(handle)
                    plot_training_results(history_dict)
                    logger.info('Τα γραφήματα εκπαίδευσης δημιουργήθηκαν επιτυχώς.\n')
                else:
                    logger.info('Ο χρήστης επέλεξε να μην προχωρήσει στην οπτικοποίηση.\n')

                return  # Τερματίζουμε τη ροή καθώς το μοντέλο φορτώθηκε και αξιολογήθηκε
            else:
                logger.info('Ο χρήστης επέλεξε να εκπαιδεύσει νέο μοντέλο. Θα αντικαταστήσει το υπάρχον.\n')

        logger.info('Ξεκινά η προεπεξεργασία των δεδομένων...\n')
        easy_train, easy_test, difficult_train, difficult_test = preprocess_data(data_path)

        logger.info('Ολοκληρώθηκε η προεπεξεργασία των δεδομένων.\n')
        logger.info(f"Μέγεθος εύκολου συνόλου εκπαίδευσης: {len(easy_train)}")
        logger.info(f"Μέγεθος εύκολου συνόλου δοκιμής: {len(easy_test)}")
        logger.info(f"Μέγεθος δύσκολου συνόλου εκπαίδευσης: {len(difficult_train)}")
        logger.info(f"Μέγεθος δύσκολου συνόλου δοκιμής: {len(difficult_test)}\n")

        train_data = pd.concat([easy_train['domain'], difficult_train['domain']])
        train_labels = pd.concat([easy_train['label'], difficult_train['label']])
        test_data = pd.concat([easy_test['domain'], difficult_test['domain']])
        test_labels = pd.concat([easy_test['label'], difficult_test['label']])

        if train_data.empty or test_data.empty:
            logger.error("Μη έγκυρα δεδομένα: Η προεπεξεργασία απέτυχε λόγω κενών δεδομένων.\n")
            print("Σφάλμα: Η προεπεξεργασία απέτυχε. Ελέγξτε τα δεδομένα εισόδου.")
            return

        logger.info('Ξεκινά η εκπαίδευση του μοντέλου...\n')
        history, tokenizer, max_length = preprocess_and_train(
            train_data.values, train_labels.values, test_data.values, test_labels.values, use_curriculum=use_curriculum
        )

        logger.info('Η εκπαίδευση του μοντέλου ολοκληρώθηκε.\n')

        logger.info('Δημιουργία γραφημάτων για τα αποτελέσματα εκπαίδευσης...\n')
        plot_training_results(history.history)
        logger.info('Τα γραφήματα εκπαίδευσης δημιουργήθηκαν επιτυχώς.\n')

        # Αποθήκευση του tokenizer, του max_length και του ιστορικού εκπαίδευσης
        save_dir = 'saved_models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_dir, 'max_length.pickle'), 'wb') as handle:
            pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_dir, 'training_history.pickle'), 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('Ο tokenizer, το max_length και το ιστορικό εκπαίδευσης αποθηκεύτηκαν επιτυχώς.\n')

        # Κωδικοποίηση και συμπλήρωση των δεδομένων δοκιμής
        test_sequences = tokenizer.texts_to_sequences(test_data.values)
        test_data_encoded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

        logger.info('Ξεκινά η αξιολόγηση του μοντέλου...\n')
        results = evaluate_and_log_model(history.model, test_data_encoded, test_labels.values)
        logger.info('Η αξιολόγηση του μοντέλου ολοκληρώθηκε.\n')

        logger.info('Δημιουργία γραφήματος Confusion Matrix...\n')
        plot_confusion_matrix(results['confusion_matrix'], ['Benign', 'Malicious'])
        logger.info('Το γράφημα Confusion Matrix δημιουργήθηκε επιτυχώς.\n')

        if use_curriculum:
            logger.info('Ξεκινά η εκπαίδευση χωρίς χρήση Curriculum Learning για σύγκριση...\n')
            non_curriculum_history, _, _ = preprocess_and_train(
                train_data.values, train_labels.values, test_data.values, test_labels.values, use_curriculum=False
            )
            logger.info('Η εκπαίδευση χωρίς Curriculum Learning ολοκληρώθηκε.\n')

            non_curriculum_results = evaluate_and_log_model(non_curriculum_history.model, test_data_encoded, test_labels.values)
            logger.info('Η αξιολόγηση του μοντέλου χωρίς Curriculum Learning ολοκληρώθηκε.\n')

            logger.info('Δημιουργία γραφήματος σύγκρισης μετρήσεων...\n')
            plot_metric_comparison(results, non_curriculum_results)
            logger.info('Το γράφημα σύγκρισης μετρήσεων δημιουργήθηκε επιτυχώς.\n')

        logger.info('Η ροή εργασιών ολοκληρώθηκε επιτυχώς.\n')

    except Exception as e:
        logger.error(f"Σφάλμα κατά την εκτέλεση της ροής εργασιών: {str(e)}\n")
        raise

if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger()

    # Έλεγχος CUDA και συσκευής για την εκπαίδευση
    device = check_device()

    # Εισαγωγή διαπιστευτηρίων από τον χρήστη με ασφαλή τρόπο
    username = input("Εισάγετε το όνομα χρήστη για το DGArchive: ").strip()
    password = input("Εισάγετε τον κωδικό πρόσβασης για το DGArchive: ").strip()

    while True:
        csv_exists, combined_data_path = check_combined_csv()
        if csv_exists:
            valid_csv = validate_and_analyze_csv(combined_data_path)
            if not valid_csv:
                user_input = input("Το αρχείο CSV δεν είναι έγκυρο. Θέλετε να συλλέξετε ξανά τα δεδομένα; (yes/no): ").strip().lower()
                if user_input == "yes":
                    collect_data(username, password)
                    continue
                else:
                    logger.info("Ο χρήστης επέλεξε να παραλείψει τη συλλογή δεδομένων.\n")
                    break
            else:
                user_input = input("Θέλετε να αντικαταστήσετε το υπάρχον αρχείο δεδομένων με νέο; (yes/no): ").strip().lower()
                if user_input == "yes":
                    collect_data(username, password)
                    continue
                else:
                    user_input = input("Θέλετε να προχωρήσετε στην προεπεξεργασία δεδομένων και εκπαίδευση; (yes/no): ").strip().lower()
                    if user_input == "yes":
                        run_pipeline(combined_data_path, use_curriculum=True)
                    else:
                        print("Η διαδικασία σταμάτησε από τον χρήστη.")
                        logger.info("Η διαδικασία σταμάτησε από τον χρήστη πριν την προεπεξεργασία δεδομένων.\n")
                    break
        else:
            print("Δεν βρέθηκε το αρχείο δεδομένων. Ξεκινά η συλλογή δεδομένων...\n")
            logger.info("Δεν βρέθηκε το αρχείο δεδομένων. Ξεκινά η συλλογή δεδομένων...\n")
            collect_data(username, password)
            continue
