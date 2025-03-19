import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "train_delay_data.csv"  # Update this with the actual file path
df = pd.read_csv("C:\\Users\\train delay data.csv")

# Display basic information
def data_overview(df):
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nFirst 5 Rows:")
    print(df.head())

data_overview(df)

# Standardizing categorical values
def standardize_columns(df):
    df['Weather Conditions'] = df['Weather Conditions'].str.strip().str.lower()
    df['Day of the Week'] = df['Day of the Week'].str.strip().str.title()
    df['Time of Day'] = df['Time of Day'].str.strip().str.title()
    df['Train Type'] = df['Train Type'].str.strip().str.title()
    df['Route Congestion'] = df['Route Congestion'].str.strip().str.title()
    return df

df = standardize_columns(df)

# Encoding 'Day of the Week'
day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
df['Day of the Week'] = df['Day of the Week'].map(day_mapping)

# Encoding 'Time of Day'
time_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
df['Time of Day'] = df['Time of Day'].map(time_mapping)

# Encoding 'Weather Conditions'
weather_mapping = {"clear": 0, "rainy": 1, "foggy": 2, "stormy": 3}
df['Weather Conditions'] = df['Weather Conditions'].map(weather_mapping)

# Encoding 'Route Congestion'
congestion_mapping = {"Low": 0, "Medium": 1, "High": 2}
df['Route Congestion'] = df['Route Congestion'].map(congestion_mapping)

# One-hot encoding 'Train Type'
df = pd.get_dummies(df, columns=['Train Type'], drop_first=True)

# Display cleaned data overview
data_overview(df)

# Save the cleaned dataset
df.to_csv("cleaned_train_delay_data.csv", index=False)
print("Cleaned dataset saved successfully!")

# Convert categorical values into numerical encoding for proper analysis

# Encoding 'Time of Day' (No need to map as we are grouping by labels)
df["Time of Day Label"] = df["Time of Day"]  # Keeping it for reference

# Step 1: Checking Evening Rush Hour Delays (5-7 PM)
avg_delay_by_time = df.groupby("Time of Day Label")["Historical Delay (min)"].mean().sort_values(ascending=False)

# Step 2: Analyzing Impact of Weather on Delays
avg_delay_by_weather = df.groupby("Weather Conditions")["Historical Delay (min)"].mean().sort_values(ascending=False)

# Step 3: Examining Route Congestion’s Contribution to Delays
avg_delay_by_congestion = df.groupby("Route Congestion")["Historical Delay (min)"].mean().sort_values(ascending=False)

# Compute % of total delays caused by High Congestion
high_congestion_delays = df[df["Route Congestion"] == 2]["Historical Delay (min)"].sum()
total_delays = df["Historical Delay (min)"].sum()
high_congestion_percentage = (high_congestion_delays / total_delays) * 100

# Ensure safe indexing for weather conditions check
if len(avg_delay_by_weather) > 1:
    weather_delay_check = avg_delay_by_weather.iloc[0] > avg_delay_by_weather.iloc[1] * 1.2
else:
    weather_delay_check = "Not enough data"



# 🔹 Key Insights from the Data
print("\n🔹 Key Insights from the Data")

# ✅ 1. Finding the Time of Day with the Highest Delays
time_labels = {0: "Morning", 1: "Afternoon", 2: "Evening", 3: "Night"}

# Get the highest and second highest delay periods
max_delay_time_code = avg_delay_by_time.idxmax()
second_max_delay_time_code = avg_delay_by_time.index[1] if len(avg_delay_by_time) > 1 else None

# Map codes back to names
max_delay_time = time_labels.get(max_delay_time_code, "Unknown")
second_max_delay_time = time_labels.get(second_max_delay_time_code, "Unknown") if second_max_delay_time_code else "No second period"

# Print insights
print(f"📌 Trains experience the **highest delays during {max_delay_time}** (~{avg_delay_by_time.max():.2f} min).")
if second_max_delay_time_code:
    print(f"📌 The second highest delays occur during **{second_max_delay_time}** (~{avg_delay_by_time.iloc[1]:.2f} min).")

# ✅ 2. Finding the Weather Condition that Causes the Longest Delays
weather_labels = {0: "Clear", 1: "Rainy", 2: "Foggy", 3: "Stormy"}

if len(avg_delay_by_weather) > 1:
    worst_weather_code = avg_delay_by_weather.idxmax()
    second_worst_weather_code = avg_delay_by_weather.index[1]

    worst_weather = weather_labels.get(worst_weather_code, "Unknown")
    second_worst_weather = weather_labels.get(second_worst_weather_code, "Unknown")

    print(f"📌 **{worst_weather} weather leads to the longest delays (~{avg_delay_by_weather.max():.2f} min).**")
    print(f"📌 This is slightly higher than delays in **{second_worst_weather} weather** (~{avg_delay_by_weather.iloc[1]:.2f} min).")
else:
    print("📌 Not enough weather data to compare the impact of different conditions.")

# ✅ 3. Evaluating the Impact of Route Congestion on Delays
low_congestion_avg = avg_delay_by_congestion.get(0, 0)
medium_congestion_avg = avg_delay_by_congestion.get(1, 0)
high_congestion_avg = avg_delay_by_congestion.get(2, 0)

print(f"📌 High congestion contributes to **~{high_congestion_percentage:.2f}%** of total delays.")

if high_congestion_avg > medium_congestion_avg > low_congestion_avg:
    print(f"📌 **Delays increase as congestion levels rise.**")
else:
    print(f"📌 **Unexpected congestion impact:** The delay pattern does not strictly increase with congestion level.")

# ✅ 4. Comparing Average Delays by Congestion Level
print(f"📌 Average delay for **low congestion**: ~{low_congestion_avg:.2f} min")
print(f"📌 Average delay for **medium congestion**: ~{medium_congestion_avg:.2f} min")
print(f"📌 Average delay for **high congestion**: ~{high_congestion_avg:.2f} min")



#plots:

# 1. Distribution of Delays
plt.figure(figsize=(8, 5))
sns.histplot(df['Historical Delay (min)'], bins=20, kde=True)
plt.xlabel('Delay (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Train Delays')
plt.show()

# 2. Delay by Day of the Week
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Day of the Week'], y=df['Historical Delay (min)'])
plt.xlabel('Day of the Week (0 = Monday, 6 = Sunday)')
plt.ylabel('Delay (minutes)')
plt.title('Train Delays by Day of the Week')
plt.show()

# 3. Delay by Time of Day
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Time of Day'], y=df['Historical Delay (min)'])
plt.xlabel('Time of Day (0 = Morning, 3 = Night)')
plt.ylabel('Delay (minutes)')
plt.title('Train Delays by Time of Day')
plt.show()

# 4. Delay by Weather Conditions
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Weather Conditions'], y=df['Historical Delay (min)'])
plt.xlabel('Weather Conditions (0 = Clear, 3 = Stormy)')
plt.ylabel('Delay (minutes)')
plt.title('Impact of Weather on Train Delays')
plt.show()

# 5. Delay by Route Congestion
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Route Congestion'], y=df['Historical Delay (min)'])
plt.xlabel('Route Congestion (0 = Low, 2 = High)')
plt.ylabel('Delay (minutes)')
plt.title('Impact of Route Congestion on Train Delays')
plt.show()

# Correlation heatmap to see relationships between variables
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()
