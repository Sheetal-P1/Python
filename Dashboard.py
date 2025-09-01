#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('pip install streamlit')


# In[3]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare data
# Example: Split into two parts
# df = pd.read_csv('8_PySolve_Cleaned_Data.csv')

#splitting the file as too large to load on Github(this code to run only once. Now the file is split)
#df.iloc[:len(df)//2].to_csv('part1.csv', index=False)
#df.iloc[len(df)//2:].to_csv('part2.csv', index=False)

#Now merge 
df1 = pd.read_csv('part1.csv')
df2 = pd.read_csv('part2.csv')
df = pd.concat([df1, df2], ignore_index=True)

#df = data.copy()

# Drop duplicate patients for demographic summary
demo = df.drop_duplicates(subset="patient_id")[[
    "patient_id", "age", "gender", "race",
    "avg_sleep_duration", "sleep_quality", "sleep_disturbance_pct", "age_group"
]]

# Sidebar filters
st.sidebar.header("Filter Patients")
selected_age_group = st.sidebar.selectbox("Age Group", demo["age_group"].unique())
selected_gender = st.sidebar.selectbox("Gender", demo["gender"].unique())

filtered_demo = demo[(demo["age_group"] == selected_age_group) & (demo["gender"] == selected_gender)]

st.title("🩺 Diabetic Patient Dashboard")
st.subheader(f"Demographics: {selected_gender} patients in {selected_age_group}")
st.dataframe(filtered_demo)

# Pie chart: Age group distribution
st.subheader("📊 Age Group Distribution")
age_group_counts = demo["age_group"].value_counts().sort_index()
fig1, ax1 = plt.subplots(figsize=(4, 4))
colors = plt.cm.Pastel1(range(len(age_group_counts)))
ax1.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
ax1.set_title('Distribution of Diabetic Patients by Age Group')
ax1.axis('equal')
st.pyplot(fig1)

# Heatmap: Correlation of numeric metrics
st.subheader("📈 Correlation Heatmap")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
fig2, ax2 = plt.subplots(figsize=(10, 6))
cax = ax2.matshow(corr_matrix, cmap='coolwarm')
fig2.colorbar(cax)
ax2.set_xticks(np.arange(len(corr_matrix.columns)))
ax2.set_yticks(np.arange(len(corr_matrix.columns)))
ax2.set_xticklabels(corr_matrix.columns, rotation=45, ha='left')
ax2.set_yticklabels(corr_matrix.columns)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', va='center', ha='center', color='black')
st.pyplot(fig2)

# Glucose trends by time of day
st.subheader("🕒 Glucose Trends by Time of Day")

# Convert and categorize time
df['glucose_test_time'] = pd.to_datetime(df['glucose_test_time'], errors='coerce')
df = df.dropna(subset=['glucose_test_time'])

def categorize_time(dt):
    hour = dt.hour
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    else:
        return 'Night'

df['time_category'] = df['glucose_test_time'].apply(categorize_time)
avg_glucose = df.groupby(['patient_id', 'time_category'])['glucose'].mean().reset_index()
time_order = {'Morning': 1, 'Afternoon': 2, 'Night': 3}
avg_glucose['time_numeric'] = avg_glucose['time_category'].map(time_order)

# Plot glucose trends
fig3, ax3 = plt.subplots(figsize=(12, 6))
patients = avg_glucose['patient_id'].unique()
colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
colors2 = plt.cm.Set3(np.linspace(0, 1, 10))
colors = np.vstack((colors1, colors2))

for i, patient in enumerate(patients):
    patient_data = avg_glucose[avg_glucose['patient_id'] == patient].sort_values('time_numeric')
    ax3.plot(patient_data['time_numeric'], patient_data['glucose'],
             label=f'{patient}', color=colors[i % len(colors)], linewidth=2, marker='o', markersize=6, alpha=0.8)

ax3.set_xticks([1, 2, 3])
ax3.set_xticklabels(['Morning', 'Afternoon', 'Night'])
ax3.set_xlabel('Time of Day')
ax3.set_ylabel('Average Glucose Level')
ax3.set_title('Glucose Trends by Time of Day and Patient')
ax3.legend(title='Patient ID', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True)
st.pyplot(fig3)

