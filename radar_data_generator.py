import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
import uuid


@dataclass
class RadarConfig:
    """Configuration for radar system parameters"""
    frequency: float = 9.375e9  # X-band radar frequency (Hz)
    pulse_width: float = 1e-6   # Pulse width (s)
    prf: float = 1000           # Pulse repetition frequency (Hz)
    antenna_gain: float = 35    # Antenna gain (dB)
    transmit_power: float = 25e3 # Transmit power (W)
    noise_figure: float = 3     # Noise figure (dB)
    range_resolution: float = 150  # Range resolution (m)
    azimuth_beamwidth: float = 1.5  # Azimuth beamwidth (degrees)
    elevation_beamwidth: float = 20  # Elevation beamwidth (degrees)


@dataclass
class EnvironmentConfig:
    """Configuration for environmental conditions"""
    sea_state: int = 3          # Sea state (0-9 scale)
    wind_speed: float = 15      # Wind speed (m/s)
    wave_height: float = 2.5    # Significant wave height (m)
    temperature: float = 15     # Air temperature (°C)
    humidity: float = 75        # Relative humidity (%)
    precipitation: float = 0    # Precipitation rate (mm/hr)


class SeaClutterGenerator:
    """Generates sea clutter returns using K-distribution and Weibull models"""
    
    def __init__(self, config: RadarConfig, env_config: EnvironmentConfig):
        self.config = config
        self.env_config = env_config
        
    def k_distribution_amplitude(self, size: int, shape: float = 2.0, scale: float = 1.0) -> np.ndarray:
        """Generate K-distributed amplitude values for sea clutter"""
        # K-distribution is modeled as product of Rayleigh and Gamma distributions
        rayleigh = np.random.rayleigh(scale, size)
        gamma_shape = shape
        gamma_scale = 1.0 / shape
        gamma_vals = np.random.gamma(gamma_shape, gamma_scale, size)
        return rayleigh * np.sqrt(gamma_vals)
    
    def weibull_clutter(self, size: int) -> np.ndarray:
        """Generate Weibull-distributed clutter amplitudes"""
        # Weibull parameters depend on sea state
        sea_state_params = {
            0: (0.5, 1.0),  # Calm sea
            1: (0.7, 1.2),
            2: (0.9, 1.5),
            3: (1.2, 2.0),  # Moderate sea
            4: (1.5, 2.5),
            5: (1.8, 3.0),
            6: (2.2, 3.5),  # Rough sea
            7: (2.5, 4.0),
            8: (2.8, 4.5),
            9: (3.0, 5.0)   # Very rough sea
        }
        
        shape, scale = sea_state_params.get(self.env_config.sea_state, (1.2, 2.0))
        return np.random.weibull(shape, size) * scale
    
    def generate_clutter_rcs(self, ranges: np.ndarray, azimuths: np.ndarray) -> np.ndarray:
        """Generate RCS values for sea clutter based on range and azimuth"""
        # Base clutter RCS decreases with range
        base_rcs = -40 - 20 * np.log10(ranges / 1000)  # dBsm
        
        # Add sea state dependency
        sea_state_factor = self.env_config.sea_state * 2
        base_rcs += sea_state_factor
        
        # Add angular dependence (backscatter varies with incidence angle)
        angular_factor = 3 * np.cos(np.radians(azimuths))
        base_rcs += angular_factor
        
        # Add random variation using K-distribution
        k_variation = self.k_distribution_amplitude(len(ranges), shape=2.0, scale=5.0)
        rcs_linear = 10 ** (base_rcs / 10) * k_variation
        
        return 10 * np.log10(rcs_linear)
    
    def generate_clutter_doppler(self, size: int) -> np.ndarray:
        """Generate Doppler values for sea clutter (mainly due to wave motion)"""
        # Sea clutter Doppler is typically low but has some spread due to wave motion
        mean_doppler = 0.0
        std_doppler = 0.5 + 0.2 * self.env_config.sea_state  # Higher sea state = more Doppler spread
        return np.random.normal(mean_doppler, std_doppler, size)


class TargetGenerator:
    """Generates vessel target returns with realistic movement patterns"""
    
    def __init__(self, config: RadarConfig):
        self.config = config
        
    def generate_vessel_rcs(self, vessel_types: List[str], ranges: np.ndarray) -> np.ndarray:
        """Generate RCS values for different vessel types"""
        # Typical RCS values for different vessel types (dBsm)
        vessel_rcs = {
            'small_boat': (5, 15),      # 5-15 dBsm
            'fishing_vessel': (15, 25),  # 15-25 dBsm
            'cargo_ship': (30, 45),     # 30-45 dBsm
            'tanker': (35, 50),         # 35-50 dBsm
            'cruise_ship': (40, 55),    # 40-55 dBsm
            'naval_vessel': (20, 35)    # 20-35 dBsm
        }
        
        rcs_values = []
        for vessel_type in vessel_types:
            min_rcs, max_rcs = vessel_rcs.get(vessel_type, (10, 20))
            base_rcs = np.random.uniform(min_rcs, max_rcs)
            
            # RCS fluctuates with aspect angle and range
            rcs_values.append(base_rcs + np.random.normal(0, 2))
        
        return np.array(rcs_values)
    
    def generate_vessel_track(self, start_pos: Tuple[float, float], 
                            duration_hours: float, 
                            vessel_type: str) -> Dict[str, List]:
        """Generate a realistic vessel track over time"""
        # Vessel speed ranges (knots)
        speed_ranges = {
            'small_boat': (5, 25),
            'fishing_vessel': (3, 15),
            'cargo_ship': (10, 20),
            'tanker': (8, 16),
            'cruise_ship': (15, 25),
            'naval_vessel': (15, 35)
        }
        
        min_speed, max_speed = speed_ranges.get(vessel_type, (10, 20))
        speed_knots = np.random.uniform(min_speed, max_speed)
        speed_ms = speed_knots * 0.514444  # Convert to m/s
        
        # Generate track with some random course changes
        start_range, start_azimuth = start_pos
        
        track = {
            'timestamps': [],
            'ranges': [],
            'azimuths': [],
            'elevations': [],
            'dopplers': [],
            'rcs_values': [],
            'snr_values': []
        }
        
        # Start time
        current_time = datetime.utcnow()
        dt = timedelta(seconds=5)  # 5-second intervals
        
        # Current position in Cartesian coordinates
        x = start_range * np.cos(np.radians(start_azimuth))
        y = start_range * np.sin(np.radians(start_azimuth))
        
        # Initial heading (random)
        heading = np.random.uniform(0, 360)
        
        total_points = int(duration_hours * 3600 / 5)  # Number of 5-second intervals
        
        for i in range(total_points):
            # Add some course variation
            if np.random.random() < 0.05:  # 5% chance of course change
                heading += np.random.normal(0, 15)  # ±15 degree variation
                heading = heading % 360
            
            # Update position
            dx = speed_ms * 5 * np.cos(np.radians(heading))  # 5-second step
            dy = speed_ms * 5 * np.sin(np.radians(heading))
            
            x += dx
            y += dy
            
            # Convert back to polar coordinates
            current_range = np.sqrt(x**2 + y**2)
            current_azimuth = np.degrees(np.arctan2(y, x))
            if current_azimuth < 0:
                current_azimuth += 360
            
            # Skip if out of radar range
            if current_range > 50000:  # 50 km max range
                break
            
            # Elevation angle (small for surface targets)
            elevation = np.random.normal(0, 0.5)
            
            # Doppler calculation (radial velocity)
            radial_velocity = speed_ms * np.cos(np.radians(heading - current_azimuth))
            doppler = radial_velocity + np.random.normal(0, 0.2)  # Add noise
            
            # RCS with fluctuation
            base_rcs = self.generate_vessel_rcs([vessel_type], np.array([current_range]))[0]
            rcs = base_rcs + np.random.normal(0, 2)
            
            # SNR calculation (simplified)
            snr = self._calculate_snr(current_range, rcs)
            
            track['timestamps'].append(current_time)
            track['ranges'].append(current_range)
            track['azimuths'].append(current_azimuth)
            track['elevations'].append(elevation)
            track['dopplers'].append(doppler)
            track['rcs_values'].append(rcs)
            track['snr_values'].append(snr)
            
            current_time += dt
        
        return track
    
    def _calculate_snr(self, range_m: float, rcs_dbsm: float) -> float:
        """Calculate Signal-to-Noise Ratio"""
        # Radar equation (simplified)
        wavelength = 3e8 / self.config.frequency
        
        # Convert RCS to linear scale
        rcs_linear = 10 ** (rcs_dbsm / 10)
        
        # Range factor (1/R^4)
        range_factor = 1 / (range_m ** 4)
        
        # Basic SNR calculation
        snr_linear = (self.config.transmit_power * self.config.antenna_gain**2 * 
                     wavelength**2 * rcs_linear * range_factor) / (64 * np.pi**3)
        
        # Add noise figure
        snr_db = 10 * np.log10(snr_linear) - self.config.noise_figure
        
        # Add some random variation
        snr_db += np.random.normal(0, 1)
        
        return snr_db


class RadarDatasetGenerator:
    """Main class for generating comprehensive radar datasets"""
    
    def __init__(self, radar_config: RadarConfig = None, env_config: EnvironmentConfig = None):
        self.radar_config = radar_config or RadarConfig()
        self.env_config = env_config or EnvironmentConfig()
        self.clutter_gen = SeaClutterGenerator(self.radar_config, self.env_config)
        self.target_gen = TargetGenerator(self.radar_config)
        
    def generate_clutter_detections(self, num_detections: int, 
                                  time_span_hours: float = 24) -> pd.DataFrame:
        """Generate sea clutter detections"""
        detections = []
        
        start_time = datetime.utcnow()
        
        for i in range(num_detections):
            # Random spatial distribution
            range_m = np.random.uniform(500, 30000)  # 0.5 to 30 km
            azimuth = np.random.uniform(0, 360)
            elevation = np.random.normal(0, 1)  # Small elevation for sea surface
            
            # Time distribution
            time_offset = np.random.uniform(0, time_span_hours * 3600)
            timestamp = start_time + timedelta(seconds=time_offset)
            
            # Generate clutter characteristics
            rcs = self.clutter_gen.generate_clutter_rcs(
                np.array([range_m]), np.array([azimuth]))[0]
            doppler = self.clutter_gen.generate_clutter_doppler(1)[0]
            
            # SNR for clutter is typically lower
            snr = np.random.normal(5, 3)  # Mean 5 dB with variation
            
            detection = {
                'TrackID': f"CLUTTER_{uuid.uuid4().hex[:8]}",
                'Range_m': range_m,
                'Azimuth_deg': azimuth,
                'Elevation_deg': elevation,
                'Doppler_ms': doppler,
                'RCS_dBsm': rcs,
                'SNR_dB': snr,
                'Timestamp': timestamp.isoformat() + 'Z',
                'Label': 'clutter'
            }
            
            detections.append(detection)
        
        return pd.DataFrame(detections)
    
    def generate_target_tracks(self, num_tracks: int, 
                             avg_track_duration_hours: float = 2) -> pd.DataFrame:
        """Generate vessel target tracks"""
        all_detections = []
        
        vessel_types = ['small_boat', 'fishing_vessel', 'cargo_ship', 
                       'tanker', 'cruise_ship', 'naval_vessel']
        
        for track_id in range(num_tracks):
            # Random vessel type
            vessel_type = np.random.choice(vessel_types)
            
            # Random starting position
            start_range = np.random.uniform(2000, 25000)  # 2-25 km
            start_azimuth = np.random.uniform(0, 360)
            
            # Track duration with variation
            duration = np.random.exponential(avg_track_duration_hours)
            duration = max(0.1, min(duration, 12))  # Limit between 6 minutes and 12 hours
            
            track = self.target_gen.generate_vessel_track(
                (start_range, start_azimuth), duration, vessel_type)
            
            # Convert track to detections
            track_id_str = f"TARGET_{track_id:06d}"
            
            for j in range(len(track['timestamps'])):
                detection = {
                    'TrackID': track_id_str,
                    'Range_m': track['ranges'][j],
                    'Azimuth_deg': track['azimuths'][j],
                    'Elevation_deg': track['elevations'][j],
                    'Doppler_ms': track['dopplers'][j],
                    'RCS_dBsm': track['rcs_values'][j],
                    'SNR_dB': track['snr_values'][j],
                    'Timestamp': track['timestamps'][j].isoformat() + 'Z',
                    'Label': 'target'
                }
                
                all_detections.append(detection)
        
        return pd.DataFrame(all_detections)
    
    def generate_complete_dataset(self, target_size_gb: float = 1.0) -> pd.DataFrame:
        """Generate a complete dataset of specified size"""
        print(f"Generating maritime radar dataset targeting {target_size_gb:.1f} GB...")
        
        # Estimate detections needed for target size
        # Rough estimate: ~100 bytes per detection
        target_detections = int(target_size_gb * 1e9 / 100)
        
        # Mix of clutter and targets (roughly 70% clutter, 30% targets)
        clutter_detections = int(target_detections * 0.7)
        target_track_count = int(target_detections * 0.3 / 50)  # Assume avg 50 detections per track
        
        print(f"Generating {clutter_detections:,} clutter detections...")
        clutter_df = self.generate_clutter_detections(clutter_detections)
        
        print(f"Generating {target_track_count:,} target tracks...")
        target_df = self.generate_target_tracks(target_track_count)
        
        # Combine datasets
        complete_df = pd.concat([clutter_df, target_df], ignore_index=True)
        
        # Sort by timestamp
        complete_df['Timestamp_dt'] = pd.to_datetime(complete_df['Timestamp'])
        complete_df = complete_df.sort_values('Timestamp_dt').drop('Timestamp_dt', axis=1)
        complete_df = complete_df.reset_index(drop=True)
        
        print(f"Dataset generated: {len(complete_df):,} total detections")
        print(f"Clutter detections: {len(complete_df[complete_df['Label'] == 'clutter']):,}")
        print(f"Target detections: {len(complete_df[complete_df['Label'] == 'target']):,}")
        
        return complete_df
    
    def save_dataset(self, dataset: pd.DataFrame, filename: str):
        """Save dataset to file"""
        if filename.endswith('.parquet'):
            dataset.to_parquet(filename, compression='gzip')
        elif filename.endswith('.csv'):
            dataset.to_csv(filename, index=False)
        else:
            # Default to parquet for better compression
            dataset.to_parquet(filename + '.parquet', compression='gzip')
        
        # Save metadata
        metadata = {
            'radar_config': self.radar_config.__dict__,
            'environment_config': self.env_config.__dict__,
            'dataset_stats': {
                'total_detections': len(dataset),
                'clutter_detections': len(dataset[dataset['Label'] == 'clutter']),
                'target_detections': len(dataset[dataset['Label'] == 'target']),
                'unique_tracks': dataset['TrackID'].nunique(),
                'time_span': {
                    'start': dataset['Timestamp'].min(),
                    'end': dataset['Timestamp'].max()
                }
            }
        }
        
        metadata_filename = filename.replace('.parquet', '').replace('.csv', '') + '_metadata.json'
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {filename}")
        print(f"Metadata saved to {metadata_filename}")


def generate_multiple_sea_states():
    """Generate datasets for different sea states"""
    sea_states = [1, 3, 6]  # Calm, moderate, rough
    datasets = {}
    
    for sea_state in sea_states:
        print(f"\n=== Generating dataset for sea state {sea_state} ===")
        
        env_config = EnvironmentConfig(sea_state=sea_state)
        generator = RadarDatasetGenerator(env_config=env_config)
        
        dataset = generator.generate_complete_dataset(target_size_gb=0.5)
        filename = f"maritime_radar_dataset_sea_state_{sea_state}.parquet"
        generator.save_dataset(dataset, filename)
        
        datasets[sea_state] = dataset
    
    return datasets


if __name__ == "__main__":
    # Generate main dataset
    generator = RadarDatasetGenerator()
    main_dataset = generator.generate_complete_dataset(target_size_gb=1.0)
    generator.save_dataset(main_dataset, "maritime_radar_dataset_main.parquet")
    
    # Generate datasets for different sea states
    additional_datasets = generate_multiple_sea_states()
    
    print("\n=== Dataset Generation Complete ===")
    print("Files generated:")
    print("- maritime_radar_dataset_main.parquet (main dataset)")
    print("- maritime_radar_dataset_sea_state_1.parquet (calm sea)")
    print("- maritime_radar_dataset_sea_state_3.parquet (moderate sea)")
    print("- maritime_radar_dataset_sea_state_6.parquet (rough sea)")
    print("- Corresponding metadata files (.json)")