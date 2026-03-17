# dga_tranco_data_collector.py

# credentials : george_betzelos  unfoldoverlaidtuesdaynegligentcarrotguacamole



import requests
import logging
import os
import pandas as pd
from requests.auth import HTTPBasicAuth
import re
import json
import sys

# Ρύθμιση του logging για καταγραφή γεγονότων σε αρχείο και στην κονσόλα
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Ρύθμιση καταγραφής σε αρχείο
file_handler = logging.FileHandler('dga_tranco_data_collector.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Ρύθμιση καταγραφής στην κονσόλα
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

# Αποτροπή διπλών καταγραφών
logger.propagate = False

def is_valid_domain(domain: str, max_length: int = 253) -> bool:
    """
    Ελέγχει εάν ένα domain είναι έγκυρο με βάση το μέγιστο επιτρεπτό μήκος και επιτρεπτούς χαρακτήρες.

    :param domain: Το domain που θα ελεγχθεί.
    :param max_length: Το μέγιστο επιτρεπτό μήκος για ένα έγκυρο domain.
    :return: True αν το domain είναι έγκυρο, False αν όχι.
    """
    if len(domain) > max_length:
        return False
    return re.match(r'^[a-zA-Z0-9\-\.]+$', domain) is not None

class DGA_TrancoDataCollector:
    def __init__(self, username: str, password: str):
        """
        Αρχικοποιεί τη συλλογή δεδομένων DGA και Tranco με πιστοποιητικά εισόδου και έξοδο αποθήκευσης.

        :param username: Όνομα χρήστη για το DGArchive.
        :param password: Κωδικός πρόσβασης για το DGArchive.
        """
        self.base_url = "https://dgarchive.caad.fkie.fraunhofer.de/today"
        self.username = username
        self.password = password
        self.save_dir = os.path.join(os.path.dirname(__file__), "combined_data")

        # Δημιουργία φακέλου αποθήκευσης αν δεν υπάρχει
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"Ο φάκελος αποθήκευσης δεδομένων δημιουργήθηκε: {self.save_dir}")
        else:
            logger.info(f"Ο φάκελος αποθήκευσης δεδομένων υπάρχει ήδη: {self.save_dir}")

        # Αρχικοποίηση συνόλου DGA για αποθήκευση μοναδικών domains
        self.dga_domains = set()

    def fetch_dga_data(self) -> None:
        """
        Ανακτά τα DGA domains για τη σημερινή ημέρα και τα αποθηκεύει για έλεγχο.
        """
        try:
            logger.info("Αίτηση για σημερινά domains με χρήση διαπιστευτηρίων...")
            response = requests.get(self.base_url, auth=HTTPBasicAuth(self.username, self.password), verify=False)

            if response.status_code == 401:
                logger.error("Αποτυχία αυθεντικοποίησης: Ελέγξτε το όνομα χρήστη και τον κωδικό.")
                return
            elif response.status_code == 200:
                logger.info("Επιτυχής αυθεντικοποίηση, λήψη απάντησης.")
            else:
                logger.warning(f"Λήψη μη αναμενόμενου κωδικού απόκρισης: {response.status_code}")

            try:
                # Ανάλυση της JSON απόκρισης και εξαγωγή των domains
                data = response.json()
                domains = []

                for key, domain_list in data.items():
                    if isinstance(domain_list, list):
                        domains.extend(domain_list)

                # Αποθήκευση όλων των δεδομένων JSON για έλεγχο
                with open(os.path.join(self.save_dir, "debug_dga_full_response.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Η πλήρης απόκριση JSON αποθηκεύτηκε για έλεγχο. Σύνολο domains: {len(domains)}")

                # Φιλτράρισμα έγκυρων domains και καταγραφή τους
                valid_domains = [domain for domain in domains if is_valid_domain(domain)]
                self.dga_domains.update(valid_domains)

                # Αποθήκευση των DGA domains σε CSV για έλεγχο
                dga_debug_file = os.path.join(self.save_dir, "debug_dga_output.csv")
                pd.DataFrame(valid_domains, columns=["domain"]).to_csv(dga_debug_file, index=False)
                logger.info(f"Τα DGA δεδομένα αποθηκεύτηκαν στο {dga_debug_file}. Έγκυρα domains: {len(valid_domains)}")
            except json.JSONDecodeError:
                logger.error("Σφάλμα στην ανάλυση της JSON απόκρισης.")
        except requests.RequestException as e:
            logger.error(f"Σφάλμα κατά την ανάκτηση των δεδομένων: {e}")

    def load_tranco_data(self) -> pd.DataFrame:
        """
        Φορτώνει τη λίστα Tranco top-1m από το αρχείο 'top-1m.csv' στον φάκελο της εφαρμογής
        όπου βρίσκεται το αρχείο main.py. Αναμένεται ότι το αρχείο περιέχει δύο στήλες χωρίς κεφαλίδα:
        'rank' για την κατάταξη και 'domain' για το όνομα του domain.

        :return: DataFrame με τα domains Tranco με ετικέτα καλοήθειας (label 0) ή κενό DataFrame αν αποτύχει η φόρτωση.
        """
        # Καθορισμός της διαδρομής του αρχείου Tranco στον ίδιο φάκελο με το αρχείο main.py
        tranco_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top-1m.csv")

        # Έλεγχος αν το αρχείο Tranco υπάρχει στην αναμενόμενη διαδρομή
        if not os.path.exists(tranco_path):
            logger.error("Το αρχείο Tranco 'top-1m.csv' δεν βρέθηκε στον ίδιο φάκελο με το main.py.")
            print("Σφάλμα: Το αρχείο Tranco 'top-1m.csv' δεν βρέθηκε στον ίδιο φάκελο με το main.py.")
            return pd.DataFrame()  # Επιστρέφει κενό DataFrame αν το αρχείο λείπει

        try:
            # Φόρτωση του αρχείου με καθορισμένα ονόματα στηλών, καθώς το αρχείο δεν περιέχει κεφαλίδα
            tranco_df = pd.read_csv(tranco_path, header=None, names=['rank', 'domain'])
            tranco_df['label'] = 0  # Ορισμός ετικέτας καλοήθειας (0)

            # Καταγραφή επιτυχούς φόρτωσης των δεδομένων και του συνολικού πλήθους domains
            logger.info(f"Επιτυχής φόρτωση δεδομένων Tranco. Σύνολο domains: {len(tranco_df)}")

            # Επικύρωση ύπαρξης της στήλης 'domain' για περαιτέρω επεξεργασία
            if 'domain' not in tranco_df.columns:
                logger.error("Η στήλη 'domain' δεν βρέθηκε στο αρχείο 'top-1m.csv'.")
                print("Σφάλμα: Η στήλη 'domain' δεν βρέθηκε στο αρχείο 'top-1m.csv'.")
                return pd.DataFrame()  # Επιστρέφει κενό DataFrame αν λείπει η στήλη

            return tranco_df  # Επιστροφή του DataFrame με τις καλοήθεις ετικέτες

        except pd.errors.ParserError as e:
            # Σφάλμα αν παρουσιαστεί πρόβλημα στην ανάλυση του αρχείου CSV
            logger.error(f"Σφάλμα κατά την ανάλυση του αρχείου Tranco: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            # Σφάλμα αν παρουσιαστεί απροσδόκητο πρόβλημα
            logger.error(f"Άγνωστο σφάλμα κατά την ανάγνωση του αρχείου Tranco: {str(e)}")
            return pd.DataFrame()





    def filter_benign_domains(self, tranco_df: pd.DataFrame) -> pd.DataFrame:
        """
        Φιλτράρει τα καλοήθη domains της Tranco λίστας για να αφαιρέσει τα κακόβουλα DGA.

        :param tranco_df: DataFrame που περιέχει τα domains Tranco.
        :return: Φιλτραρισμένο DataFrame με καλοήθη domains.
        """
        initial_count = len(tranco_df)
        filtered_df = tranco_df[~tranco_df['domain'].isin(self.dga_domains)]
        final_count = len(filtered_df)
        logger.info(f"Φιλτραρίστηκαν {initial_count - final_count} domains από την Tranco λίστα ως κακόβουλα.")
        return filtered_df

    def save_combined_data(self, benign_df: pd.DataFrame):
        """
        Συνδυάζει τα DGA domains και τα φιλτραρισμένα Tranco δεδομένα σε ένα αρχείο CSV.

        :param benign_df: DataFrame με καλοήθη domains από Tranco.
        """
        dga_df = pd.DataFrame({'domain': list(self.dga_domains), 'label': 1})  # Ετικέτα κακόβουλου (1)
        combined_df = pd.concat([dga_df, benign_df[['domain', 'label']]], ignore_index=True).drop_duplicates(subset='domain')
        combined_filename = os.path.join(self.save_dir, "combined_dga_tranco_data.csv")
        combined_df.to_csv(combined_filename, index=False)
        logger.info(f"Το συνδυασμένο αρχείο αποθηκεύτηκε στο {combined_filename}. Σύνολο domains: {len(combined_df)}")

    def save_dga_data(self):
        """
        Αποθηκεύει μόνο τα DGA domains σε ξεχωριστό αρχείο CSV.
        """
        if not self.dga_domains:
            logger.warning("Δεν υπάρχουν DGA domains για αποθήκευση.")
            return

        dga_filename = os.path.join(self.save_dir, "dga_data.csv")
        dga_df = pd.DataFrame({'domain': list(self.dga_domains), 'label': 1})
        dga_df.to_csv(dga_filename, index=False, encoding='utf-8')
        logger.info(f"DGA data αποθηκεύτηκαν στο {dga_filename}. Σύνολο DGA domains: {len(dga_df)}")

    def run(self):
        """
        Εκτελεί τη διαδικασία συλλογής, φιλτραρίσματος και αποθήκευσης δεδομένων.
        """
        logger.info("Ξεκινά η διαδικασία συλλογής δεδομένων...")
        self.fetch_dga_data()  # Ανάκτηση DGA δεδομένων
        tranco_df = self.load_tranco_data()  # Φόρτωση της λίστας Tranco
        filtered_benign_df = self.filter_benign_domains(tranco_df)  # Φιλτράρισμα καλοήθων domains
        self.save_combined_data(filtered_benign_df)  # Αποθήκευση συνδυασμένων δεδομένων
        self.save_dga_data()  # Αποθήκευση μόνο των DGA domains
        logger.info("Η διαδικασία συλλογής και αποθήκευσης ολοκληρώθηκε επιτυχώς.")


if __name__ == "__main__":
    username = input("Εισάγετε το όνομα χρήστη για το DGArchive: ").strip()
    password = input("Εισάγετε τον κωδικό πρόσβασης για το DGArchive: ").strip()
    collector = DGA_TrancoDataCollector(username=username, password=password)
    collector.run()
