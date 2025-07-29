import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import joblib
import warnings
warnings.filterwarnings('ignore')


class RadarFeatureEngineering:
    """Feature engineering for radar detection classification"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw radar data"""
        df_features = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if 'Timestamp' in df_features.columns:
            df_features['Timestamp_dt'] = pd.to_datetime(df_features['Timestamp'], format='ISO8601')
            
            # Time-based features
            df_features['Hour'] = df_features['Timestamp_dt'].dt.hour
            df_features['DayOfWeek'] = df_features['Timestamp_dt'].dt.dayofweek
            df_features['TimeOfDay'] = df_features['Hour'].apply(self._get_time_period)
        
        # Spatial features
        df_features['Range_km'] = df_features['Range_m'] / 1000
        df_features['Range_bin'] = pd.cut(df_features['Range_m'], 
                                        bins=[0, 5000, 15000, 30000, np.inf], 
                                        labels=['short', 'medium', 'long', 'very_long'])
        
        # Convert range bins to numeric
        df_features['Range_bin_encoded'] = LabelEncoder().fit_transform(df_features['Range_bin'])
        
        # Azimuth sectors
        df_features['Azimuth_sector'] = pd.cut(df_features['Azimuth_deg'], 
                                             bins=8, labels=False)  # 8 sectors of 45 degrees each
        
        # Doppler features
        df_features['Doppler_abs'] = np.abs(df_features['Doppler_ms'])
        df_features['Doppler_category'] = df_features['Doppler_abs'].apply(self._categorize_doppler)
        df_features['Doppler_category_encoded'] = LabelEncoder().fit_transform(df_features['Doppler_category'])
        
        # Signal strength features
        df_features['RCS_linear'] = 10 ** (df_features['RCS_dBsm'] / 10)
        df_features['SNR_linear'] = 10 ** (df_features['SNR_dB'] / 10)
        
        # Power-to-noise ratio
        df_features['Power_ratio'] = df_features['RCS_linear'] / (df_features['SNR_linear'] + 1e-10)
        
        # Range-normalized RCS
        df_features['RCS_range_normalized'] = df_features['RCS_dBsm'] + 40 * np.log10(df_features['Range_km'])
        
        # Track-based features (if TrackID is available)
        if 'TrackID' in df_features.columns:
            track_features = self._create_track_features(df_features)
            df_features = df_features.merge(track_features, on='TrackID', how='left')
        
        # Statistical features combinations
        df_features['RCS_SNR_ratio'] = df_features['RCS_dBsm'] / (df_features['SNR_dB'] + 1e-10)
        df_features['Range_Doppler_product'] = df_features['Range_km'] * df_features['Doppler_abs']
        df_features['Elevation_abs'] = np.abs(df_features['Elevation_deg'])
        
        # Radar cross-section stability (variance indicator)
        df_features['RCS_stability'] = 1 / (1 + df_features['RCS_dBsm'].var())
        
        return df_features
    
    def _get_time_period(self, hour: int) -> str:
        """Categorize time of day"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'
    
    def _categorize_doppler(self, doppler_abs: float) -> str:
        """Categorize Doppler velocity"""
        if doppler_abs < 1:
            return 'stationary'
        elif doppler_abs < 5:
            return 'slow'
        elif doppler_abs < 15:
            return 'medium'
        else:
            return 'fast'
    
    def _create_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create track-level aggregated features"""
        track_stats = df.groupby('TrackID').agg({
            'Range_m': ['mean', 'std', 'min', 'max'],
            'Azimuth_deg': ['mean', 'std'],
            'Elevation_deg': ['mean', 'std'],
            'Doppler_ms': ['mean', 'std', 'min', 'max'],
            'RCS_dBsm': ['mean', 'std', 'min', 'max'],
            'SNR_dB': ['mean', 'std'],
            'TrackID': 'count'  # Number of detections per track
        }).reset_index()
        
        # Flatten column names
        track_stats.columns = ['TrackID'] + [f'track_{col[0]}_{col[1]}' for col in track_stats.columns[1:]]
        track_stats.rename(columns={'track_TrackID_count': 'track_length'}, inplace=True)
        
        # Additional track features
        # Track consistency (low std indicates consistent target)
        track_stats['track_consistency'] = (track_stats['track_RCS_dBsm_std'] + 
                                          track_stats['track_Doppler_ms_std']) / 2
        
        # Track mobility
        track_stats['track_mobility'] = track_stats['track_Doppler_ms_std']
        
        return track_stats
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Label') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and target vector"""
        # Create engineered features
        df_features = self.create_features(df)
        
        # Select numeric features for modeling
        numeric_features = [
            'Range_m', 'Azimuth_deg', 'Elevation_deg', 'Doppler_ms', 'RCS_dBsm', 'SNR_dB',
            'Range_km', 'Range_bin_encoded', 'Azimuth_sector', 'Doppler_abs', 
            'Doppler_category_encoded', 'RCS_linear', 'SNR_linear', 'Power_ratio',
            'RCS_range_normalized', 'RCS_SNR_ratio', 'Range_Doppler_product', 'Elevation_abs'
        ]
        
        # Add time features if available
        if 'Hour' in df_features.columns:
            numeric_features.extend(['Hour', 'DayOfWeek'])
        
        # Add track features if available
        track_features = [col for col in df_features.columns if col.startswith('track_')]
        if track_features:
            numeric_features.extend(track_features)
        
        # Filter features that actually exist in the dataframe
        available_features = [f for f in numeric_features if f in df_features.columns]
        
        # Extract features and target
        X = df_features[available_features].fillna(0)  # Fill NaN with 0
        y = self.label_encoder.fit_transform(df_features[target_col])
        
        return X.values, y, available_features


class RadarTargetClassifier:
    """Wrapper class for radar target classification models"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = RadarFeatureEngineering()
        self.feature_names = []
        self.is_fitted = False
        
    def train_models(self, df: pd.DataFrame, test_size: float = 0.2, 
                    random_state: int = 42) -> Dict[str, Any]:
        """Train both XGBoost and Random Forest models"""
        
        print("Preparing features...")
        X, y, feature_names = self.feature_engineer.prepare_features(df)
        self.feature_names = feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # Train XGBoost
        print("\nTraining XGBoost model...")
        xgb_model = self._train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
        self.models['xgboost'] = xgb_model
        results['xgboost'] = self._evaluate_model(xgb_model, X_test_scaled, y_test, 'XGBoost')
        
        # Train Random Forest
        print("\nTraining Random Forest model...")
        rf_model = self._train_random_forest(X_train, y_train, X_test, y_test)  # RF doesn't need scaling
        self.models['random_forest'] = rf_model
        results['random_forest'] = self._evaluate_model(rf_model, X_test, y_test, 'Random Forest')
        
        # Store preprocessing objects
        self.scaler = scaler
        self.X_test = X_test_scaled if 'xgboost' in self.models else X_test
        self.y_test = y_test
        
        self.is_fitted = True
        
        # Feature importance analysis
        self._analyze_feature_importance()
        
        return results
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost model with hyperparameter tuning"""
        
        # Define parameter grid for GridSearch
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
        )
        
        # Grid search with cross-validation
        print("Performing hyperparameter tuning for XGBoost...")
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='f1', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best XGBoost parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest model with hyperparameter tuning"""
        
        # Define parameter grid for GridSearch
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create Random Forest classifier
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        print("Performing hyperparameter tuning for Random Forest...")
        grid_search = GridSearchCV(
            rf_model, param_grid, cv=3, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best Random Forest parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                       model_name: str) -> Dict[str, Any]:
        """Evaluate model performance"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=['Clutter', 'Target'], 
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Clutter', 'Target'])}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def _analyze_feature_importance(self):
        """Analyze and plot feature importance for both models"""
        
        plt.figure(figsize=(15, 10))
        
        # XGBoost feature importance
        if 'xgboost' in self.models:
            plt.subplot(2, 2, 1)
            xgb_importance = self.models['xgboost'].feature_importances_
            indices = np.argsort(xgb_importance)[::-1][:15]  # Top 15 features
            
            plt.title('XGBoost Feature Importance (Top 15)')
            plt.bar(range(len(indices)), xgb_importance[indices])
            plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            plt.subplot(2, 2, 2)
            rf_importance = self.models['random_forest'].feature_importances_
            indices = np.argsort(rf_importance)[::-1][:15]  # Top 15 features
            
            plt.title('Random Forest Feature Importance (Top 15)')
            plt.bar(range(len(indices)), rf_importance[indices])
            plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
        
        # ROC Curves
        plt.subplot(2, 2, 3)
        for model_name, model in self.models.items():
            if model_name == 'xgboost':
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            else:  # random_forest
                # Need to use unscaled data for RF
                X_test_rf = self.scaler.inverse_transform(self.X_test) if hasattr(self, 'scaler') else self.X_test
                y_pred_proba = model.predict_proba(X_test_rf)[:, 1]
            
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        # Precision-Recall Curves
        plt.subplot(2, 2, 4)
        for model_name, model in self.models.items():
            if model_name == 'xgboost':
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            else:  # random_forest
                X_test_rf = self.scaler.inverse_transform(self.X_test) if hasattr(self, 'scaler') else self.X_test
                y_pred_proba = model.predict_proba(X_test_rf)[:, 1]
            
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            plt.plot(recall, precision, label=f'{model_name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single(self, detection_data: Dict[str, float], model_name: str = 'xgboost') -> Dict[str, Any]:
        """Predict classification for a single detection"""
        
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.models.keys())}")
        
        # Convert to DataFrame
        df_single = pd.DataFrame([detection_data])
        
        # Prepare features
        X_single, _, _ = self.feature_engineer.prepare_features(df_single, target_col=None)
        
        # Scale if using XGBoost
        if model_name == 'xgboost' and hasattr(self, 'scaler'):
            X_single = self.scaler.transform(X_single)
        
        model = self.models[model_name]
        
        # Prediction
        prediction = model.predict(X_single)[0]
        prediction_proba = model.predict_proba(X_single)[0]
        
        # Convert back to labels
        label = 'target' if prediction == 1 else 'clutter'
        confidence = prediction_proba[prediction]
        
        return {
            'prediction': label,
            'confidence': confidence,
            'clutter_probability': prediction_proba[0],
            'target_probability': prediction_proba[1]
        }
    
    def save_models(self, filepath_prefix: str = 'radar_classifier'):
        """Save trained models and preprocessing objects"""
        
        if not self.is_fitted:
            raise ValueError("Models must be trained before saving")
        
        # Save models
        for model_name, model in self.models.items():
            filename = f"{filepath_prefix}_{model_name}.joblib"
            joblib.dump(model, filename)
            print(f"Saved {model_name} model to {filename}")
        
        # Save preprocessing objects
        preprocessing_data = {
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names
        }
        
        preprocessing_filename = f"{filepath_prefix}_preprocessing.joblib"
        joblib.dump(preprocessing_data, preprocessing_filename)
        print(f"Saved preprocessing objects to {preprocessing_filename}")
    
    def load_models(self, filepath_prefix: str = 'radar_classifier'):
        """Load trained models and preprocessing objects"""
        
        # Load preprocessing objects
        preprocessing_filename = f"{filepath_prefix}_preprocessing.joblib"
        preprocessing_data = joblib.load(preprocessing_filename)
        
        self.scaler = preprocessing_data['scaler']
        self.feature_engineer = preprocessing_data['feature_engineer']
        self.feature_names = preprocessing_data['feature_names']
        
        # Load models
        model_files = {
            'xgboost': f"{filepath_prefix}_xgboost.joblib",
            'random_forest': f"{filepath_prefix}_random_forest.joblib"
        }
        
        self.models = {}
        for model_name, filename in model_files.items():
            try:
                self.models[model_name] = joblib.load(filename)
                print(f"Loaded {model_name} model from {filename}")
            except FileNotFoundError:
                print(f"Model file {filename} not found, skipping...")
        
        self.is_fitted = len(self.models) > 0
        
        return self.is_fitted


def cross_validate_models(df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
    """Perform cross-validation on both models"""
    
    feature_engineer = RadarFeatureEngineering()
    X, y, feature_names = feature_engineer.prepare_features(df)
    
    # Scale features for XGBoost
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # XGBoost cross-validation
    xgb_model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.1, n_estimators=200,
        subsample=0.9, colsample_bytree=0.9,
        objective='binary:logistic', random_state=42
    )
    
    xgb_scores = cross_val_score(xgb_model, X_scaled, y, cv=cv_folds, scoring='f1')
    results['xgboost'] = {
        'mean_f1': xgb_scores.mean(),
        'std_f1': xgb_scores.std(),
        'scores': xgb_scores
    }
    
    # Random Forest cross-validation
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', random_state=42
    )
    
    rf_scores = cross_val_score(rf_model, X, y, cv=cv_folds, scoring='f1')
    results['random_forest'] = {
        'mean_f1': rf_scores.mean(),
        'std_f1': rf_scores.std(),
        'scores': rf_scores
    }
    
    print("Cross-Validation Results:")
    for model_name, scores in results.items():
        print(f"{model_name}: F1 = {scores['mean_f1']:.4f} (+/- {scores['std_f1']:.4f})")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Radar Target Classification Model Training")
    print("=========================================")
    
    # This would be used with the generated dataset
    # df = pd.read_parquet('maritime_radar_dataset_main.parquet')
    # classifier = RadarTargetClassifier()
    # results = classifier.train_models(df)
    # classifier.save_models()
    
    print("Module loaded successfully. Use with generated radar dataset.")