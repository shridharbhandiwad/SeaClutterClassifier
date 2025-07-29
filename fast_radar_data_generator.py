import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from typing import Tuple, List, Dict, Any
import uuid
from dataclasses import dataclass
import time

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

class FastRadarDataGenerator:
    """Optimized radar data generator for large datasets"""
    
    def __init__(self, radar_config: RadarConfig = None, env_config: EnvironmentConfig = None):
        self.radar_config = radar_config or RadarConfig()
        self.env_config = env_config or EnvironmentConfig()
        
    def generate_sea_clutter_batch(self, num_detections: int) -> pd.DataFrame:
        """Generate sea clutter detections in batch - optimized for speed"""
        print(f"Generating {num_detections:,} sea clutter detections...")
        
        # Generate all spatial coordinates at once
        ranges = np.random.uniform(500, 30000, num_detections)  # 0.5 to 30 km
        azimuths = np.random.uniform(0, 360, num_detections)
        elevations = np.random.normal(0, 1, num_detections)  # Small elevation for sea surface
        
        # Generate time distribution
        start_time = datetime.utcnow()
        time_offsets = np.random.uniform(0, 24 * 3600, num_detections)  # 24 hours span
        timestamps = [start_time + timedelta(seconds=offset) for offset in time_offsets]
        
        # Vectorized RCS calculation for sea clutter
        base_rcs = -40 - 20 * np.log10(ranges / 1000)  # dBsm, decreases with range
        sea_state_factor = self.env_config.sea_state * 2
        base_rcs += sea_state_factor
        
        # Angular dependence (backscatter varies with incidence angle)
        angular_factor = 3 * np.cos(np.radians(azimuths))
        base_rcs += angular_factor
        
        # K-distribution variation (vectorized)
        shape = 2.0
        scale = 5.0
        rayleigh_vals = np.random.rayleigh(scale, num_detections)
        gamma_vals = np.random.gamma(shape, 1.0/shape, num_detections)
        k_variation = rayleigh_vals * np.sqrt(gamma_vals)
        
        rcs_linear = 10 ** (base_rcs / 10) * k_variation
        rcs_values = 10 * np.log10(rcs_linear)
        
        # Vectorized Doppler generation
        mean_doppler = 0.0
        std_doppler = 0.5 + 0.2 * self.env_config.sea_state
        doppler_values = np.random.normal(mean_doppler, std_doppler, num_detections)
        
        # SNR for clutter (vectorized)
        snr_values = np.random.normal(5, 3, num_detections)  # Mean 5 dB with variation
        
        # Generate track IDs
        track_ids = [f"CLUTTER_{uuid.uuid4().hex[:8]}" for _ in range(num_detections)]
        
        # Create DataFrame efficiently
        data = {
            'TrackID': track_ids,
            'Range_m': ranges,
            'Azimuth_deg': azimuths,
            'Elevation_deg': elevations,
            'Doppler_ms': doppler_values,
            'RCS_dBsm': rcs_values,
            'SNR_dB': snr_values,
            'Timestamp': [ts.isoformat() + 'Z' for ts in timestamps],
            'Label': ['clutter'] * num_detections
        }
        
        return pd.DataFrame(data)
    
    def generate_vessel_tracks_batch(self, num_target_detections: int, num_tracks: int = None) -> pd.DataFrame:
        """Generate vessel target tracks in batch - optimized for speed"""
        
        if num_tracks is None:
            # Estimate reasonable number of tracks (avg 50 detections per track)
            num_tracks = max(1, num_target_detections // 50)
        
        print(f"Generating {num_tracks:,} vessel tracks for {num_target_detections:,} target detections...")
        
        all_detections = []
        vessel_types = ['small_boat', 'fishing_vessel', 'cargo_ship', 'tanker', 'cruise_ship', 'naval_vessel']
        
        # Vessel speed ranges (m/s)
        speed_ranges = {
            'small_boat': (5*0.514444, 25*0.514444),
            'fishing_vessel': (3*0.514444, 15*0.514444),
            'cargo_ship': (10*0.514444, 20*0.514444),
            'tanker': (8*0.514444, 16*0.514444),
            'cruise_ship': (15*0.514444, 25*0.514444),
            'naval_vessel': (15*0.514444, 35*0.514444)
        }
        
        # Vessel RCS ranges (dBsm)
        vessel_rcs = {
            'small_boat': (5, 15),
            'fishing_vessel': (15, 25),
            'cargo_ship': (30, 45),
            'tanker': (35, 50),
            'cruise_ship': (40, 55),
            'naval_vessel': (20, 35)
        }
        
        detections_per_track = num_target_detections // num_tracks
        remaining_detections = num_target_detections % num_tracks
        
        for track_id in range(num_tracks):
            # Add extra detection to some tracks to reach exact target count
            track_detections = detections_per_track + (1 if track_id < remaining_detections else 0)
            
            if track_detections == 0:
                continue
                
            # Random vessel type and parameters
            vessel_type = np.random.choice(vessel_types)
            min_speed, max_speed = speed_ranges[vessel_type]
            speed_ms = np.random.uniform(min_speed, max_speed)
            
            # Starting position
            start_range = np.random.uniform(2000, 25000)
            start_azimuth = np.random.uniform(0, 360)
            start_heading = np.random.uniform(0, 360)
            
            # Convert to Cartesian for easier tracking
            x = start_range * np.cos(np.radians(start_azimuth))
            y = start_range * np.sin(np.radians(start_azimuth))
            
            # Generate time series for this track
            start_time = datetime.utcnow()
            dt_seconds = 5  # 5-second intervals
            
            # Pre-generate course changes
            course_changes = np.random.random(track_detections) < 0.05  # 5% chance
            heading_changes = np.random.normal(0, 15, track_detections)
            
            # Initialize arrays for this track
            ranges_track = np.zeros(track_detections)
            azimuths_track = np.zeros(track_detections)
            elevations_track = np.random.normal(0, 0.5, track_detections)
            dopplers_track = np.zeros(track_detections)
            timestamps_track = []
            
            current_heading = start_heading
            current_x, current_y = x, y
            
            for i in range(track_detections):
                # Update heading if course change
                if course_changes[i]:
                    current_heading += heading_changes[i]
                    current_heading = current_heading % 360
                
                # Update position
                dx = speed_ms * dt_seconds * np.cos(np.radians(current_heading))
                dy = speed_ms * dt_seconds * np.sin(np.radians(current_heading))
                current_x += dx
                current_y += dy
                
                # Convert back to polar
                current_range = np.sqrt(current_x**2 + current_y**2)
                current_azimuth = np.degrees(np.arctan2(current_y, current_x))
                if current_azimuth < 0:
                    current_azimuth += 360
                
                # Skip if out of range
                if current_range > 50000:
                    break
                
                ranges_track[i] = current_range
                azimuths_track[i] = current_azimuth
                
                # Doppler calculation
                radial_velocity = speed_ms * np.cos(np.radians(current_heading - current_azimuth))
                dopplers_track[i] = radial_velocity + np.random.normal(0, 0.2)
                
                # Timestamp
                timestamps_track.append(start_time + timedelta(seconds=i * dt_seconds))
            
            # Trim arrays to actual length (in case track ended early)
            valid_count = len(timestamps_track)
            ranges_track = ranges_track[:valid_count]
            azimuths_track = azimuths_track[:valid_count]
            elevations_track = elevations_track[:valid_count]
            dopplers_track = dopplers_track[:valid_count]
            
            # Generate RCS values (vectorized)
            min_rcs, max_rcs = vessel_rcs[vessel_type]
            base_rcs = np.random.uniform(min_rcs, max_rcs)
            rcs_values = base_rcs + np.random.normal(0, 2, valid_count)
            
            # Calculate SNR (vectorized)
            wavelength = 3e8 / self.radar_config.frequency
            rcs_linear = 10 ** (rcs_values / 10)
            range_factor = 1 / (ranges_track ** 4)
            snr_linear = (self.radar_config.transmit_power * self.radar_config.antenna_gain**2 * 
                         wavelength**2 * rcs_linear * range_factor) / (64 * np.pi**3)
            snr_values = 10 * np.log10(snr_linear) - self.radar_config.noise_figure
            snr_values += np.random.normal(0, 1, valid_count)
            
            # Create track detections
            track_id_str = f"TARGET_{track_id:06d}"
            
            for i in range(valid_count):
                detection = {
                    'TrackID': track_id_str,
                    'Range_m': ranges_track[i],
                    'Azimuth_deg': azimuths_track[i],
                    'Elevation_deg': elevations_track[i],
                    'Doppler_ms': dopplers_track[i],
                    'RCS_dBsm': rcs_values[i],
                    'SNR_dB': snr_values[i],
                    'Timestamp': timestamps_track[i].isoformat() + 'Z',
                    'Label': 'target'
                }
                all_detections.append(detection)
        
        return pd.DataFrame(all_detections)
    
    def generate_100k_dataset(self, clutter_ratio: float = 0.7) -> pd.DataFrame:
        """Generate exactly 100,000 datapoints with specified clutter/target ratio"""
        
        total_detections = 100000
        clutter_detections = int(total_detections * clutter_ratio)
        target_detections = total_detections - clutter_detections
        
        print(f"🎯 Generating exactly {total_detections:,} datapoints")
        print(f"   - Sea clutter: {clutter_detections:,} detections ({clutter_ratio:.1%})")
        print(f"   - Sea tracks: {target_detections:,} detections ({1-clutter_ratio:.1%})")
        
        start_time = time.time()
        
        # Generate clutter detections
        clutter_df = self.generate_sea_clutter_batch(clutter_detections)
        clutter_time = time.time() - start_time
        print(f"✅ Clutter generation completed in {clutter_time:.2f} seconds")
        
        # Generate target detections
        target_start = time.time()
        target_df = self.generate_vessel_tracks_batch(target_detections)
        target_time = time.time() - target_start
        print(f"✅ Target generation completed in {target_time:.2f} seconds")
        
        # Combine datasets
        combine_start = time.time()
        complete_df = pd.concat([clutter_df, target_df], ignore_index=True)
        
        # Sort by timestamp for realism
        complete_df['Timestamp_dt'] = pd.to_datetime(complete_df['Timestamp'])
        complete_df = complete_df.sort_values('Timestamp_dt').drop('Timestamp_dt', axis=1)
        complete_df = complete_df.reset_index(drop=True)
        
        combine_time = time.time() - combine_start
        total_time = time.time() - start_time
        
        print(f"✅ Dataset combination completed in {combine_time:.2f} seconds")
        print(f"🚀 Total generation time: {total_time:.2f} seconds")
        print(f"📊 Final dataset: {len(complete_df):,} detections")
        print(f"   - Clutter: {len(complete_df[complete_df['Label'] == 'clutter']):,}")
        print(f"   - Targets: {len(complete_df[complete_df['Label'] == 'target']):,}")
        print(f"   - Unique tracks: {complete_df['TrackID'].nunique():,}")
        
        return complete_df
    
    def save_dataset(self, dataset: pd.DataFrame, filename: str = "radar_100k_dataset", 
                    compression: str = 'gzip') -> None:
        """Save dataset with metadata"""
        
        # Ensure .parquet extension
        if not filename.endswith('.parquet'):
            filename += '.parquet'
        
        print(f"💾 Saving dataset to {filename}...")
        
        try:
            dataset.to_parquet(filename, compression=compression, index=False)
            print(f"✅ Dataset saved successfully")
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            # Fallback to CSV
            csv_filename = filename.replace('.parquet', '.csv')
            dataset.to_csv(csv_filename, index=False)
            print(f"💾 Fallback: saved as CSV to {csv_filename}")
        
        # Save metadata
        metadata = {
            'generation_timestamp': datetime.utcnow().isoformat(),
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
                },
                'feature_columns': list(dataset.columns),
                'memory_usage_mb': dataset.memory_usage(deep=True).sum() / (1024**2)
            }
        }
        
        metadata_filename = filename.replace('.parquet', '_metadata.json').replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadata saved to {metadata_filename}")
        
        # Print dataset summary
        print(f"\n📈 Dataset Summary:")
        print(f"   Size: {dataset.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        print(f"   Columns: {list(dataset.columns)}")
        print(f"   Date range: {dataset['Timestamp'].min()} to {dataset['Timestamp'].max()}")


def main():
    """Generate 100K radar dataset quickly"""
    
    print("🚀 Fast Radar Data Generator - 100K Dataset")
    print("=" * 50)
    
    # Initialize generator with default configs
    generator = FastRadarDataGenerator()
    
    # Generate 100K dataset
    dataset = generator.generate_100k_dataset(clutter_ratio=0.7)
    
    # Save dataset
    generator.save_dataset(dataset, "radar_100k_dataset")
    
    print("\n🎉 Generation complete!")
    print("Ready for machine learning training.")


if __name__ == "__main__":
    main()