import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any
import json
from datetime import datetime
import os

# Import our custom modules
from radar_ml_models import RadarTargetClassifier
from radar_data_generator import RadarDatasetGenerator, RadarConfig, EnvironmentConfig


class InteractiveRadarTester:
    """Interactive web interface for testing radar classification models"""
    
    def __init__(self):
        self.classifier = RadarTargetClassifier()
        self.model_loaded = False
        
    def load_models(self, model_path_prefix: str = 'radar_classifier'):
        """Load pre-trained models"""
        try:
            self.model_loaded = self.classifier.load_models(model_path_prefix)
            return self.model_loaded
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def create_input_form(self) -> Dict[str, float]:
        """Create input form for radar detection parameters"""
        st.header("🎯 Radar Detection Parameters")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spatial Parameters")
            range_m = st.number_input(
                "Range (meters)", 
                min_value=500.0, 
                max_value=50000.0, 
                value=10000.0,
                step=100.0,
                help="Distance from radar to detection point"
            )
            
            azimuth_deg = st.number_input(
                "Azimuth (degrees)", 
                min_value=0.0, 
                max_value=360.0, 
                value=180.0,
                step=1.0,
                help="Bearing angle from radar (0° = North)"
            )
            
            elevation_deg = st.number_input(
                "Elevation (degrees)", 
                min_value=-10.0, 
                max_value=10.0, 
                value=0.0,
                step=0.1,
                help="Elevation angle (typically near 0° for surface targets)"
            )
        
        with col2:
            st.subheader("Signal Parameters")
            doppler_ms = st.number_input(
                "Doppler Velocity (m/s)", 
                min_value=-50.0, 
                max_value=50.0, 
                value=5.0,
                step=0.1,
                help="Radial velocity component (positive = approaching)"
            )
            
            rcs_dbsm = st.number_input(
                "RCS (dBsm)", 
                min_value=-50.0, 
                max_value=60.0, 
                value=20.0,
                step=1.0,
                help="Radar Cross Section in decibels square meters"
            )
            
            snr_db = st.number_input(
                "SNR (dB)", 
                min_value=-10.0, 
                max_value=50.0, 
                value=15.0,
                step=1.0,
                help="Signal-to-Noise Ratio in decibels"
            )
        
        # Additional parameters
        st.subheader("Additional Information")
        col3, col4 = st.columns(2)
        
        with col3:
            track_id = st.text_input(
                "Track ID", 
                value="USER_INPUT_001",
                help="Unique identifier for this detection"
            )
        
        with col4:
            timestamp = st.text_input(
                "Timestamp (ISO format)", 
                value=datetime.utcnow().isoformat() + 'Z',
                help="Timestamp in ISO 8601 format"
            )
        
        # Preset scenarios
        st.subheader("📋 Quick Preset Scenarios")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("🚢 Large Vessel"):
                return self._get_large_vessel_preset()
        
        with preset_col2:
            if st.button("🛥️ Small Boat"):
                return self._get_small_boat_preset()
        
        with preset_col3:
            if st.button("🌊 Sea Clutter"):
                return self._get_sea_clutter_preset()
        
        return {
            'TrackID': track_id,
            'Range_m': range_m,
            'Azimuth_deg': azimuth_deg,
            'Elevation_deg': elevation_deg,
            'Doppler_ms': doppler_ms,
            'RCS_dBsm': rcs_dbsm,
            'SNR_dB': snr_db,
            'Timestamp': timestamp,
            'Label': 'unknown'  # Will be predicted
        }
    
    def _get_large_vessel_preset(self) -> Dict[str, Any]:
        """Preset for large vessel detection"""
        return {
            'TrackID': 'PRESET_LARGE_VESSEL',
            'Range_m': 15000.0,
            'Azimuth_deg': 45.0,
            'Elevation_deg': 0.2,
            'Doppler_ms': 8.5,
            'RCS_dBsm': 35.0,
            'SNR_dB': 25.0,
            'Timestamp': datetime.utcnow().isoformat() + 'Z',
            'Label': 'unknown'
        }
    
    def _get_small_boat_preset(self) -> Dict[str, Any]:
        """Preset for small boat detection"""
        return {
            'TrackID': 'PRESET_SMALL_BOAT',
            'Range_m': 5000.0,
            'Azimuth_deg': 120.0,
            'Elevation_deg': -0.1,
            'Doppler_ms': 12.0,
            'RCS_dBsm': 8.0,
            'SNR_dB': 12.0,
            'Timestamp': datetime.utcnow().isoformat() + 'Z',
            'Label': 'unknown'
        }
    
    def _get_sea_clutter_preset(self) -> Dict[str, Any]:
        """Preset for sea clutter detection"""
        return {
            'TrackID': 'PRESET_SEA_CLUTTER',
            'Range_m': 8000.0,
            'Azimuth_deg': 200.0,
            'Elevation_deg': 0.0,
            'Doppler_ms': 0.3,
            'RCS_dBsm': -25.0,
            'SNR_dB': 3.0,
            'Timestamp': datetime.utcnow().isoformat() + 'Z',
            'Label': 'unknown'
        }
    
    def predict_and_display(self, detection_data: Dict[str, Any]):
        """Make prediction and display results"""
        if not self.model_loaded:
            st.error("❌ Models not loaded. Please load models first.")
            return
        
        st.header("🔮 Classification Results")
        
        # Make predictions with both models
        results = {}
        for model_name in self.classifier.models.keys():
            try:
                result = self.classifier.predict_single(detection_data, model_name)
                results[model_name] = result
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")
        
        if not results:
            st.error("No predictions could be made.")
            return
        
        # Display results
        col1, col2 = st.columns(2)
        
        for i, (model_name, result) in enumerate(results.items()):
            with col1 if i == 0 else col2:
                st.subheader(f"🤖 {model_name.title().replace('_', ' ')}")
                
                # Prediction with confidence
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Color based on prediction
                color = "🎯" if prediction == "target" else "🌊"
                
                st.metric(
                    "Prediction", 
                    f"{color} {prediction.title()}", 
                    f"{confidence:.1%} confidence"
                )
                
                # Probability breakdown
                st.write("**Probability Breakdown:**")
                prob_df = pd.DataFrame({
                    'Class': ['Clutter', 'Target'],
                    'Probability': [result['clutter_probability'], result['target_probability']]
                })
                
                # Create probability bar chart
                fig = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    title=f"{model_name.title()} Probabilities",
                    color='Class',
                    color_discrete_map={'Clutter': '#ff7f0e', 'Target': '#1f77b4'}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        if len(results) > 1:
            st.subheader("📊 Model Comparison")
            
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Model': model_name.title().replace('_', ' '),
                    'Prediction': result['prediction'].title(),
                    'Confidence': result['confidence'],
                    'Target Probability': result['target_probability']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Agreement indicator
            predictions = [r['prediction'] for r in results.values()]
            if len(set(predictions)) == 1:
                st.success("✅ Both models agree on the classification!")
            else:
                st.warning("⚠️ Models disagree on classification.")
    
    def create_radar_plot(self, detection_data: Dict[str, Any]):
        """Create radar plot visualization"""
        st.subheader("📡 Radar Visualization")
        
        # Convert to polar coordinates for plotting
        range_km = detection_data['Range_m'] / 1000
        azimuth_rad = np.radians(detection_data['Azimuth_deg'])
        
        # Create polar plot
        fig = go.Figure()
        
        # Add detection point
        fig.add_trace(go.Scatterpolar(
            r=[range_km],
            theta=[detection_data['Azimuth_deg']],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Detection',
            text=[f"Range: {range_km:.1f} km<br>"
                  f"Azimuth: {detection_data['Azimuth_deg']:.1f}°<br>"
                  f"Doppler: {detection_data['Doppler_ms']:.1f} m/s<br>"
                  f"RCS: {detection_data['RCS_dBsm']:.1f} dBsm"],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add range rings
        range_rings = [5, 10, 15, 20, 25, 30]
        for ring in range_rings:
            if ring <= max(50, range_km * 1.2):
                fig.add_trace(go.Scatterpolar(
                    r=[ring] * 360,
                    theta=list(range(360)),
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(50, range_km * 1.2)],
                    title="Range (km)"
                ),
                angularaxis=dict(
                    direction="clockwise",
                    start=90,
                    title="Azimuth (degrees)"
                )
            ),
            title="Radar Detection Position",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_parameter_analysis(self, detection_data: Dict[str, Any]):
        """Create parameter analysis visualization"""
        st.subheader("📈 Parameter Analysis")
        
        # Create parameter comparison with typical ranges
        parameters = {
            'Range (km)': {
                'value': detection_data['Range_m'] / 1000,
                'typical_clutter': (0.5, 30),
                'typical_target': (2, 25)
            },
            'RCS (dBsm)': {
                'value': detection_data['RCS_dBsm'],
                'typical_clutter': (-50, -20),
                'typical_target': (5, 50)
            },
            'SNR (dB)': {
                'value': detection_data['SNR_dB'],
                'typical_clutter': (-5, 10),
                'typical_target': (10, 40)
            },
            'Doppler (m/s)': {
                'value': abs(detection_data['Doppler_ms']),
                'typical_clutter': (0, 2),
                'typical_target': (2, 30)
            }
        }
        
        fig = go.Figure()
        
        for i, (param_name, param_data) in enumerate(parameters.items()):
            value = param_data['value']
            clutter_range = param_data['typical_clutter']
            target_range = param_data['typical_target']
            
            # Add clutter range
            fig.add_trace(go.Scatter(
                x=[clutter_range[0], clutter_range[1]],
                y=[i, i],
                mode='lines',
                line=dict(color='orange', width=10),
                name='Typical Clutter Range' if i == 0 else '',
                showlegend=i == 0,
                legendgroup='clutter'
            ))
            
            # Add target range
            fig.add_trace(go.Scatter(
                x=[target_range[0], target_range[1]],
                y=[i + 0.1, i + 0.1],
                mode='lines',
                line=dict(color='blue', width=10),
                name='Typical Target Range' if i == 0 else '',
                showlegend=i == 0,
                legendgroup='target'
            ))
            
            # Add actual value
            fig.add_trace(go.Scatter(
                x=[value],
                y=[i + 0.05],
                mode='markers',
                marker=dict(color='red', size=12, symbol='diamond'),
                name='Your Value' if i == 0 else '',
                showlegend=i == 0,
                legendgroup='actual'
            ))
        
        fig.update_layout(
            title="Parameter Comparison with Typical Ranges",
            xaxis_title="Parameter Value",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(parameters))),
                ticktext=list(parameters.keys())
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Maritime Radar Target Classifier",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Maritime Radar Target Classifier")
    st.markdown("**Interactive Testing Interface for Radar Target vs. Clutter Classification**")
    
    # Initialize the tester
    tester = InteractiveRadarTester()
    
    # Sidebar for model management
    with st.sidebar:
        st.header("🔧 Model Management")
        
        # Model loading
        if st.button("🔄 Load Models"):
            with st.spinner("Loading models..."):
                if tester.load_models():
                    st.success("✅ Models loaded successfully!")
                    st.write(f"Available models: {list(tester.classifier.models.keys())}")
                else:
                    st.error("❌ Failed to load models")
        
        # Model status
        if tester.model_loaded:
            st.success("🟢 Models Ready")
        else:
            st.warning("🟡 Models Not Loaded")
        
        st.markdown("---")
        
        # Information
        st.header("ℹ️ About")
        st.markdown("""
        This tool allows you to test radar classification models 
        that distinguish between:
        - 🌊 **Sea Clutter**: Background noise from waves
        - 🎯 **Targets**: Actual vessels and objects
        
        **Input Parameters:**
        - **Range**: Distance from radar
        - **Azimuth**: Bearing angle
        - **Elevation**: Vertical angle
        - **Doppler**: Radial velocity
        - **RCS**: Radar cross-section
        - **SNR**: Signal-to-noise ratio
        """)
    
    # Main content area
    if not tester.model_loaded:
        st.info("👈 Please load the models from the sidebar to begin testing.")
        return
    
    # Input form
    detection_data = tester.create_input_form()
    
    # Test button
    if st.button("🎯 Classify Detection", type="primary"):
        with st.spinner("Analyzing radar detection..."):
            # Make predictions
            tester.predict_and_display(detection_data)
            
            # Create visualizations
            col1, col2 = st.columns(2)
            with col1:
                tester.create_radar_plot(detection_data)
            with col2:
                tester.create_parameter_analysis(detection_data)
    
    # Batch testing option
    st.markdown("---")
    st.subheader("📊 Batch Testing")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with multiple detections for batch testing",
        type=['csv'],
        help="CSV should contain columns: Range_m, Azimuth_deg, Elevation_deg, Doppler_ms, RCS_dBsm, SNR_dB"
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} detections")
            
            if st.button("🚀 Run Batch Classification"):
                with st.spinner("Processing batch..."):
                    results = []
                    
                    for idx, row in batch_df.iterrows():
                        detection = row.to_dict()
                        if 'TrackID' not in detection:
                            detection['TrackID'] = f"BATCH_{idx:04d}"
                        if 'Timestamp' not in detection:
                            detection['Timestamp'] = datetime.utcnow().isoformat() + 'Z'
                        
                        # Get prediction from primary model
                        primary_model = list(tester.classifier.models.keys())[0]
                        pred = tester.classifier.predict_single(detection, primary_model)
                        
                        results.append({
                            'Index': idx,
                            'TrackID': detection.get('TrackID', f"BATCH_{idx}"),
                            'Prediction': pred['prediction'],
                            'Confidence': pred['confidence'],
                            'Target_Probability': pred['target_probability']
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.subheader("Batch Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Detections", len(results_df))
                    with col2:
                        targets = len(results_df[results_df['Prediction'] == 'target'])
                        st.metric("Predicted Targets", targets)
                    with col3:
                        clutter = len(results_df[results_df['Prediction'] == 'clutter'])
                        st.metric("Predicted Clutter", clutter)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "batch_classification_results.csv",
                        "text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing batch file: {str(e)}")


if __name__ == "__main__":
    main()