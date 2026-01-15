import os
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    'SEED': 42,
    'N_SAMPLES': 50000,
    'FRAUD_RATIO': 0.01,
    'EMBEDDING_DIM': 5,
    'EPOCHS': 20,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.001,
    'ANOMALY_PERCENTILE': 0.995,  
    'OUTPUT_DIR': 'audit_reports'
}

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [FORENSIC_AI] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{CONFIG['OUTPUT_DIR']}/system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(CONFIG['SEED'])


class GLDataGenerator:
    """
    Simulates enterprise General Ledger data with injected specific fraud patterns.
    """
    def __init__(self, n_samples=50000, fraud_ratio=0.01):
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio

    def generate(self):
        logger.info(f"Generating synthetic GL with {self.n_samples} transactions...")
        
        n_normal = int(self.n_samples * (1 - self.fraud_ratio))
        
        data_config = [
            {'dept': 'Sales',    'acct': 'Revenue', 'users': ['User_S1', 'User_S2']},
            {'dept': 'IT',       'acct': 'Expense', 'users': ['User_I1', 'User_I2']},
            {'dept': 'HR',       'acct': 'Expense', 'users': ['User_H1']},
            {'dept': 'Finance',  'acct': 'Liability', 'users': ['User_F1', 'User_F2']},
            {'dept': 'Ops',      'acct': 'Asset',     'users': ['User_O1']}
        ]
        
        rows = []
        for _ in range(n_normal):
            scenario = random.choice(data_config)
            rows.append({
                'UserID': random.choice(scenario['users']),
                'Department': scenario['dept'],
                'AccountType': scenario['acct'],
                'Amount': np.random.lognormal(mean=6, sigma=1),
                'Label': 'Normal'
            })
            
        n_fraud = self.n_samples - n_normal
        logger.info(f"Injecting {n_fraud} fraud scenarios...")
        
        for _ in range(n_fraud // 2):
            rows.append({
                'UserID': 'User_I1', 
                'Department': 'IT',
                'AccountType': 'Asset', 
                'Amount': np.random.uniform(8000, 9500), 
                'Label': 'Fraud_Scenario_A'
            })
            
        for _ in range(n_fraud - (n_fraud // 2)):
            rows.append({
                'UserID': 'User_S1',
                'Department': 'Sales',
                'AccountType': 'Expense', 
                'Amount': np.random.uniform(2000, 4000),
                'Label': 'Fraud_Scenario_B'
            })

        df = pd.DataFrame(rows)
        df = df.sample(frac=1).reset_index(drop=True)
        return df


class ForensicPreprocessor:
    """
    Handles scaling and label encoding, keeping track of mappings for decoding later.
    """
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.cat_cols = ['UserID', 'Department', 'AccountType']
        
    def fit_transform(self, df):
        df_encoded = df.copy()
        
        for col in self.cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            self.encoders[col] = le
            
        df_encoded['Amount'] = self.scaler.fit_transform(df[['Amount']])
        
        return df_encoded
    
    def get_vocab_sizes(self):
        return {col: len(self.encoders[col].classes_) for col in self.cat_cols}
    
    def inverse_transform(self, df_encoded):
        """Reverts data back to human readable format for reporting"""
        df_decoded = df_encoded.copy()
        for col in self.cat_cols:
            df_decoded[col] = self.encoders[col].inverse_transform(df_encoded[col])
        
        return df_decoded


class ForensicAutoencoder(keras.Model):
    """
    A Hybrid Autoencoder that handles Categorical Embeddings and Numerics.
    It learns to reconstruct the 'Embedding Space' of valid transactions.
    """
    def __init__(self, vocab_sizes, embedding_dim=5):
        super(ForensicAutoencoder, self).__init__()
        self.vocab_sizes = vocab_sizes
        
        self.emb_user = layers.Embedding(vocab_sizes['UserID'], embedding_dim, name='emb_user')
        self.emb_dept = layers.Embedding(vocab_sizes['Department'], embedding_dim, name='emb_dept')
        self.emb_acct = layers.Embedding(vocab_sizes['AccountType'], embedding_dim, name='emb_acct')
        
        self.flatten = layers.Flatten()
        self.concat = layers.Concatenate()
        
        self.encoder = models.Sequential([
            layers.Dense(32, activation="elu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation="elu"),
            layers.Dense(8, activation="linear", name="bottleneck") 
        ])
        
        target_dim = (3 * embedding_dim) + 1
        self.decoder = models.Sequential([
            layers.Dense(16, activation="elu"),
            layers.Dense(32, activation="elu"),
            layers.Dense(target_dim, activation="linear", name="reconstruction")
        ])
        
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs):
        user_idx = inputs['UserID']
        dept_idx = inputs['Department']
        acct_idx = inputs['AccountType']
        amount = inputs['Amount']
        
        x_user = self.flatten(self.emb_user(user_idx))
        x_dept = self.flatten(self.emb_dept(dept_idx))
        x_acct = self.flatten(self.emb_acct(acct_idx))
        
        feature_vector = self.concat([x_user, x_dept, x_acct, amount])
        
        encoded = self.encoder(feature_vector)
        reconstructed = self.decoder(encoded)
        
        return reconstructed, feature_vector

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]
        
        with tf.GradientTape() as tape:
            reconstructed, feature_vector = self(data)
            loss = tf.reduce_mean(tf.square(feature_vector - reconstructed))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def main():
    logger.info(">>> STARTING FORENSIC AUDIT PROTOCOL <<<")
    
    gen = GLDataGenerator(n_samples=CONFIG['N_SAMPLES'], fraud_ratio=CONFIG['FRAUD_RATIO'])
    df_raw = gen.generate()
    
    df_report = df_raw.copy()
    
    processor = ForensicPreprocessor()
    df_proc = processor.fit_transform(df_raw)
    
    x_train = {
        'UserID': df_proc['UserID'].values,
        'Department': df_proc['Department'].values,
        'AccountType': df_proc['AccountType'].values,
        'Amount': df_proc['Amount'].values.reshape(-1, 1).astype('float32')
    }
    
    vocab_sizes = processor.get_vocab_sizes()
    model = ForensicAutoencoder(vocab_sizes, embedding_dim=CONFIG['EMBEDDING_DIM'])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']))
    
    logger.info("Training Neural Network on General Ledger...")
    history = model.fit(
        x_train, 
        epochs=CONFIG['EPOCHS'], 
        batch_size=CONFIG['BATCH_SIZE'],
        verbose=1,
        shuffle=True
    )
    
    logger.info("Scoring transactions for anomalies...")
    reconstructed, feature_vector = model.predict(x_train)
    
    mse = np.mean(np.power(feature_vector - reconstructed, 2), axis=1)
    
    df_report['Score'] = mse
    
    threshold = np.quantile(mse, CONFIG['ANOMALY_PERCENTILE'])
    logger.info(f"Anomaly Threshold (MSE): {threshold:.4f}")
    
    df_report['Is_Anomaly'] = df_report['Score'] > threshold
    
    anomalies = df_report[df_report['Is_Anomaly'] == True].sort_values('Score', ascending=False)
    
    print("\n" + "="*60)
    print(f"FORENSIC AUDIT SUMMARY")
    print("="*60)
    print(f"Total Transactions: {len(df_report)}")
    print(f"Anomalies Flagged:  {len(anomalies)}")
    print(f"Percentage Flagged: {(len(anomalies)/len(df_report))*100:.2f}%")
    print("-" * 60)
    print("TOP 10 MOST SUSPICIOUS ENTRIES:")
    print(anomalies[['UserID', 'Department', 'AccountType', 'Amount', 'Label', 'Score']].head(10))
    print("="*60)
    
    save_path = f"{CONFIG['OUTPUT_DIR']}/audit_results.csv"
    anomalies.to_csv(save_path, index=False)
    logger.info(f"Full anomaly report saved to {save_path}")
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df_report['Score'], bins=100, kde=False, color='blue', alpha=0.6, label='Normal Traffic')
    sns.histplot(anomalies['Score'], bins=20, kde=False, color='red', alpha=1.0, label='Flagged Anomalies')
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    plt.yscale('log') 
    plt.title("Forensic Autoencoder: Reconstruction Error Distribution (Log Scale)")
    plt.xlabel("Anomaly Score (MSE)")
    plt.ylabel("Frequency (Log)")
    plt.legend()
    plt.savefig(f"{CONFIG['OUTPUT_DIR']}/anomaly_distribution.png")
    logger.info("Visualization saved.")
    plt.show()

if __name__ == "__main__":
    main()