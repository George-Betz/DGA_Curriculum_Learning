import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import os

# Ρύθμιση του logger
logger = logging.getLogger('model_training')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('model_training.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


# ✅ Helper to safely set learning rate
def safe_set_lr(model, lr):
    """Safely updates or recreates optimizer with new learning rate."""
    try:
        if hasattr(model.optimizer, 'learning_rate') and hasattr(model.optimizer.learning_rate, 'assign'):
            model.optimizer.learning_rate.assign(lr)
        else:
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=model.optimizer, loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        logger.info(f"Learning rate successfully set to {lr}")
    except Exception as e:
        logger.warning(f"Failed to update learning rate dynamically, recompiling optimizer. Error: {e}")
        model.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=model.optimizer, loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])


def build_model(input_length: int, vocab_size: int) -> Sequential:
    """Κατασκευή LSTM μοντέλου για ανάλυση χαρακτηριστικών domain."""
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=128, input_length=input_length),
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(64, dropout=0.3),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    logger.info('✅ Το μοντέλο κατασκευάστηκε επιτυχώς.')
    return model


def tokenize_domains(train_domains: np.ndarray, test_domains: np.ndarray):
    """Κωδικοποιεί τα domains σε ακολουθίες χαρακτήρων."""
    tokenizer = Tokenizer(char_level=True, oov_token='UNK')
    tokenizer.fit_on_texts(train_domains)
    train_sequences = tokenizer.texts_to_sequences(train_domains)
    test_sequences = tokenizer.texts_to_sequences(test_domains)

    max_length = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in test_sequences))
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    logger.info('✅ Η κωδικοποίηση και συμπλήρωση των domains ολοκληρώθηκε επιτυχώς.')
    return train_padded, test_padded, tokenizer, max_length


def calculate_difficulty(domains):
    """Υπολογίζει τη δυσκολία κάθε domain με βάση το μήκος και τον αριθμό μοναδικών χαρακτήρων."""
    difficulties = []
    for d in domains:
        length_score = len(d)
        unique_score = len(set(d))
        # Weight: 70% length + 30% unique charu8i
        # acters
        difficulty = 0.7 * length_score + 0.3 * unique_score
        difficulties.append(difficulty)
    return np.array(difficulties)


def preprocess_and_train(train_data: np.ndarray, train_labels: np.ndarray,
                         test_data: np.ndarray, test_labels: np.ndarray,
                         use_curriculum: bool = False,
                         total_epochs: int = 20):
    """Πλήρης διαδικασία εκπαίδευσης μοντέλου με ή χωρίς Curriculum Learning."""
    # Tokenization
    train_data_encoded, test_data_encoded, tokenizer, input_length = tokenize_domains(train_data, test_data)
    vocab_size = len(tokenizer.word_index)
    model = build_model(input_length, vocab_size)

    # ✅ Compute class weights to handle imbalance
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights = dict(zip(classes, weights))

    logger.info(f"Κατανομή ετικετών: {Counter(train_labels)}")
    logger.info(f"Class weights: {class_weights}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    if use_curriculum:
        logger.info("🚀 Εκπαίδευση με χρήση Curriculum Learning.")
        difficulties = calculate_difficulty(train_data)
        sorted_indices = np.argsort(difficulties)
        train_data_encoded = train_data_encoded[sorted_indices]
        train_labels = train_labels[sorted_indices]

        total_size = len(train_data_encoded)
        stages = 4
        history = None

        # Distribute total epochs across curriculum stages (total ≈ total_epochs)
        stage_ratios = np.array([0.15, 0.20, 0.25, 0.40])
        stage_epochs_list = np.round(stage_ratios * total_epochs).astype(int)

        for stage in range(1, stages + 1):
            end_idx = int(total_size * (stage / stages))
            subset_data = train_data_encoded[:end_idx]
            subset_labels = train_labels[:end_idx]

            # Add some harder samples for robustness
            if stage > 1:
                hard_indices = np.random.choice(np.arange(end_idx, total_size),
                                                size=min(50, total_size - end_idx),
                                                replace=False)
                subset_data = np.concatenate([subset_data, train_data_encoded[hard_indices]])
                subset_labels = np.concatenate([subset_labels, train_labels[hard_indices]])

            lr = 1e-3 * (0.5 ** (stage - 1))
            safe_set_lr(model, lr)

            stage_epochs = stage_epochs_list[stage - 1]
            logger.info(f"Στάδιο {stage}/{stages} - {end_idx} δείγματα, epochs={stage_epochs}, lr={lr:.5f}")

            history = model.fit(
                subset_data,
                subset_labels,
                validation_data=(test_data_encoded, test_labels),
                epochs=stage_epochs,
                batch_size=32,
                verbose=1,
                callbacks=callbacks,
                class_weight=class_weights  # ✅ Apply balancing
            )

        # 🔹 Final fine-tuning on full data
        logger.info("🔧 Τελικό στάδιο fine-tuning σε όλο το σύνολο δεδομένων.")
        safe_set_lr(model, 1e-4)
        history = model.fit(
            train_data_encoded,
            train_labels,
            validation_data=(test_data_encoded, test_labels),
            epochs=max(2, int(total_epochs * 0.1)),
            batch_size=32,
            verbose=1,
            callbacks=callbacks,
            class_weight=class_weights
        )

    else:
        logger.info("🧠 Εκπαίδευση χωρίς Curriculum Learning.")
        history = model.fit(
            train_data_encoded,
            train_labels,
            validation_data=(test_data_encoded, test_labels),
            epochs=20,
            batch_size=32,
            verbose=1,
            # callbacks=callbacks,
            class_weight=class_weights
        )

    logger.info("✅ Η προεπεξεργασία και εκπαίδευση του μοντέλου ολοκληρώθηκε επιτυχώς.")

    # Save model
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'dga_detection_model.keras'))
    logger.info(f"💾 Το μοντέλο αποθηκεύτηκε στο {os.path.join(save_dir, 'dga_detection_model.keras')}")

    return history, tokenizer, input_length


if __name__ == "__main__":
    # Παράδειγμα δεδομένων (για δοκιμή)
    train_domains = np.array([
        "abc.com", "maliciousdomain123.com", "benign.com", "test.net",
        "freevpnserver.ru", "update1234.biz", "paypal-secure-login.net"
    ])
    test_domains = np.array(["xyz.com", "badsite.com", "safesite.org", "securemailupdate.io"])
    train_labels = np.array([0, 1, 0, 0, 1, 1, 1])
    test_labels = np.array([0, 1, 0, 1])

    history, tokenizer, max_length = preprocess_and_train(
        train_domains,
        train_labels,
        test_domains,
        test_labels,
        use_curriculum=True,
        total_epochs=12
    )
