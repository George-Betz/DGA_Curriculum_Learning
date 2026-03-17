# model_training.py

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt

# Ρύθμιση του logger για καταγραφή σφαλμάτων και πληροφοριών
logger = logging.getLogger('model_training')
logger.setLevel(logging.INFO)

# Προσθήκη handler για καταγραφή σε αρχείο
file_handler = logging.FileHandler('model_training.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# -----------------------------
# Build the LSTM model
# -----------------------------
def build_model(input_length: int, vocab_size: int) -> Sequential:
    """
    Κατασκευή νευρωνικού δικτύου LSTM με embedding layer για ανάλυση χαρακτηριστικών domain.
    """
    try:
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size + 1, output_dim=64, input_length=input_length))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(64))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info('Το μοντέλο κατασκευάστηκε επιτυχώς.')
        return model
    except Exception as e:
        logger.error(f'Σφάλμα κατά την κατασκευή του μοντέλου: {str(e)}')
        raise


# -----------------------------
# Tokenize domains
# -----------------------------
def tokenize_domains(train_domains: np.ndarray, test_domains: np.ndarray):
    """
    Κωδικοποιεί τα domains σε ακολουθίες χαρακτήρων για χρήση στο embedding layer.
    """
    try:
        tokenizer = Tokenizer(char_level=True, oov_token='UNK')
        tokenizer.fit_on_texts(train_domains)
        train_sequences = tokenizer.texts_to_sequences(train_domains)
        test_sequences = tokenizer.texts_to_sequences(test_domains)

        max_length = max(
            max(len(seq) for seq in train_sequences),
            max(len(seq) for seq in test_sequences)
        )

        train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
        test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

        logger.info('Η κωδικοποίηση και συμπλήρωση των domains ολοκληρώθηκε επιτυχώς.')
        return train_padded, test_padded, tokenizer, max_length

    except Exception as e:
        logger.error(f'Σφάλμα κατά την κωδικοποίηση των domains: {str(e)}')
        raise


# -----------------------------
# Curriculum Learning training
# -----------------------------
def preprocess_and_train(train_data: np.ndarray,
                         train_labels: np.ndarray,
                         test_data: np.ndarray,
                         test_labels: np.ndarray,
                         use_curriculum: bool = False):
    """
    Πλήρης διαδικασία για κωδικοποίηση δεδομένων, κατασκευή και εκπαίδευση μοντέλου.
    """
    try:
        # Κωδικοποίηση των domains
        train_data_encoded, test_data_encoded, tokenizer, input_length = tokenize_domains(train_data, test_data)
        vocab_size = len(tokenizer.word_index)
        model = build_model(input_length, vocab_size)

        # Callbacks για σταθερότητα
        callbacks = [
            EarlyStopping(patience=2, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)
        ]

        # --------------------------------------------------------
        # Με Curriculum Learning (βελτιωμένη έκδοση)
        # --------------------------------------------------------
        if use_curriculum:
            logger.info("Ξεκινά η εκπαίδευση με χρήση Curriculum Learning (βελτιωμένη).")

            # Υπολογισμός εντροπίας (χαρακτηριστικό δυσκολίας)
            def compute_entropy(domain):
                counts = Counter(domain)
                probs = np.array(list(counts.values())) / len(domain)
                return -np.sum(probs * np.log2(probs))

            entropies = np.array([compute_entropy(d) for d in train_data])

            # Τρία στάδια με αύξηση δυσκολίας
            thresholds = np.percentile(entropies, [30, 60, 90])
            total_epochs = 12
            stage_epochs = total_epochs // (len(thresholds) + 1)

            for i, thr in enumerate(thresholds):
                indices = np.where(entropies <= thr)[0]
                stage_data = train_data_encoded[indices]
                stage_labels = train_labels[indices]

                logger.info(f"Στάδιο {i+1}: Εκπαίδευση με {len(indices)} δείγματα (εντροπία ≤ {thr:.2f})")

                model.fit(
                    stage_data,
                    stage_labels,
                    epochs=stage_epochs,
                    batch_size=32,
                    validation_data=(test_data_encoded, test_labels),
                    callbacks=callbacks
                )

                val_loss, val_acc = model.evaluate(test_data_encoded, test_labels, verbose=0)
                logger.info(f"Ακρίβεια επαλήθευσης μετά το στάδιο {i+1}: {val_acc:.4f}")

            # Τελική φάση με όλα τα δεδομένα
            logger.info("Τελικό στάδιο εκπαίδευσης με όλα τα δεδομένα.")
            history = model.fit(
                train_data_encoded,
                train_labels,
                epochs=stage_epochs,
                batch_size=32,
                validation_data=(test_data_encoded, test_labels),
                callbacks=callbacks
            )

        # --------------------------------------------------------
        # Χωρίς Curriculum Learning (baseline)
        # --------------------------------------------------------
        else:
            logger.info("Ξεκινά η εκπαίδευση χωρίς Curriculum Learning.")
            history = model.fit(
                train_data_encoded,
                train_labels,
                validation_data=(test_data_encoded, test_labels),
                epochs=12,
                batch_size=32,
                callbacks=callbacks
            )

        # --------------------------------------------------------
        # Αποθήκευση του μοντέλου
        # --------------------------------------------------------
        logger.info("Η προεπεξεργασία και εκπαίδευση του μοντέλου ολοκληρώθηκε επιτυχώς.")

        save_dir = 'saved_models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_name = 'dga_detection_model_curriculum.keras' if use_curriculum else 'dga_detection_model_baseline.keras'
        model.save(os.path.join(save_dir, model_name))
        logger.info(f"Το μοντέλο αποθηκεύτηκε στο {os.path.join(save_dir, model_name)}")

        return history, tokenizer, input_length

    except Exception as e:
        logger.error(f'Σφάλμα κατά την προεπεξεργασία και εκπαίδευση: {str(e)}')
        raise

if __name__ == "__main__":
    # Τα παρακάτω είναι παραδείγματα και θα πρέπει να αντικατασταθούν με τα πραγματικά  δεδομένα
    train_domains = np.array(["abc.com", "maliciousdomain123.com", "benign.com", "test.net"])
    test_domains = np.array(["xyz.com", "badsite.com", "safesite.org"])
    train_labels = np.array([0, 1, 0, 0])
    test_labels = np.array([0, 1, 0])
    
    history, tokenizer, max_length = preprocess_and_train(train_domains, train_labels, test_domains, test_labels)

