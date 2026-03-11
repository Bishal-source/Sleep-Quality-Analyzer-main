import pandas as pd
import numpy as np

np.random.seed(42)

def generate_data(n_samples, quality):
    data = []

    for _ in range(n_samples):
        if quality == "Good":
            # Good sleep - healthy parameters with some variation
            sleep_duration = np.random.uniform(6.5, 9.5)
            stress_level = np.random.randint(1, 5)
            physical_activity = np.random.randint(45, 150)
            caffeine_intake = np.random.uniform(0, 2.5)
            screen_time = np.random.uniform(0, 3)
            spo2_before = np.random.randint(95, 100)
            spo2_after = np.random.randint(96, 100)
            heart_rate_before = np.random.randint(58, 80)
            heart_rate_after = np.random.randint(52, 75)

        elif quality == "Average":
            # Average sleep - moderate parameters, overlaps with Good/Bad
            # Sleep duration can be slightly less or slightly more
            sleep_duration = np.random.choice([
                np.random.uniform(5.5, 6.5),   # Slightly short
                np.random.uniform(9.5, 10.5)   # Slightly long
            ])
            stress_level = np.random.randint(3, 8)  # Can overlap
            physical_activity = np.random.randint(20, 70)  # Moderate
            caffeine_intake = np.random.uniform(1.5, 4.5)
            screen_time = np.random.uniform(2, 5.5)
            spo2_before = np.random.randint(92, 97)
            spo2_after = np.random.randint(93, 98)
            heart_rate_before = np.random.randint(65, 88)
            heart_rate_after = np.random.randint(60, 88)

        else:  # Bad
            # Bad sleep - unhealthy parameters
            sleep_duration = np.random.choice([
                np.random.uniform(3, 5.5),   # Too short
                np.random.uniform(10.5, 12)  # Too long
            ])
            stress_level = np.random.randint(5, 10)
            physical_activity = np.random.randint(0, 40)
            caffeine_intake = np.random.uniform(3, 8)
            screen_time = np.random.uniform(4, 10)
            spo2_before = np.random.randint(88, 94)
            spo2_after = np.random.randint(89, 95)
            heart_rate_before = np.random.randint(70, 100)
            heart_rate_after = np.random.randint(65, 100)

        # Add small random noise to make it more realistic
        sleep_duration += np.random.uniform(-0.3, 0.3)
        caffeine_intake += np.random.uniform(-0.3, 0.3)
        screen_time += np.random.uniform(-0.3, 0.3)

        data.append([
            round(sleep_duration, 1),
            stress_level,
            physical_activity,
            round(caffeine_intake, 1),
            round(screen_time, 1),
            spo2_before,
            spo2_after,
            heart_rate_before,
            heart_rate_after,
            quality
        ])

    return data

# Generate 10,000 samples total (balanced)
print("Generating dataset...")

samples_per_category = 10000 // 3
good_data = generate_data(samples_per_category, "Good")
average_data = generate_data(samples_per_category, "Average")
bad_data = generate_data(samples_per_category, "Bad")

# Combine all data
all_data = good_data + average_data + bad_data

# Shuffle the data
np.random.shuffle(all_data)

# Create DataFrame
df = pd.DataFrame(all_data, columns=[
    "Sleep Duration",
    "Stress Level",
    "Physical Activity",
    "Caffeine Intake",
    "Screen Time",
    "SpO2 Before",
    "SpO2 After",
    "Heart Rate Before",
    "Heart Rate After",
    "Sleep Quality"
])

# Save to CSV
df.to_csv("dataset/sleep_quality_dataset.csv", index=False)

print(f"Dataset created with {len(df)} rows")
print(f"\nClass distribution:")
print(df["Sleep Quality"].value_counts())
print(f"\nSample statistics:")
print(df.describe())
print(f"\nDataset saved to: dataset/sleep_quality_dataset.csv")