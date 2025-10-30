import pandas as pd
import time



# Function to get text description for a patient
def get_patient_description(patient_values):
    try:
        completion = model.completions.create(
            messages=[
                {"role": "user", 
                 "content": "Here is a patient current information (if this patient can use diagnostic equipment or has access to healthcare facilities):\n"
                            "The variables include gender, age, glucose, hematocrit, creatinine, sodium, blood urea nitrogen, hemoglobin, heart rate, mean blood pressure, platelets, respiratory rate, bicarbonate, red blood cell count, and anion gap\n"
                            f"The values are {','.join(map(str, patient_values))}.\n"
                            "How would this patient describe his/her symptoms via text and describe his/her feeling without any diagnostic measurements? Only output the description text."
            }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error getting description: {e}")
        return None

# Read data

PATH = 'dataset/'
df = pd.read_csv(PATH)

# Process rows and save descriptions immediately
# Open file in append mode
with open(PATH + 'MIMIC_ALL_text.txt', 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        patient_id = index + 1  # Adding 1 because index starts at 0
        values = row.values.tolist()
        # print("The values are ", values)
        print(f"Processing patient {patient_id}...")
        description = get_patient_description(values)
        
        if description:
            # Write description immediately after generation
            f.write(f"Patient ID: {patient_id}\n{description}\n{'='*50}\n")
            f.flush()  # Ensure writing to disk
            # print(f"Description for patient {patient_id}:\n{description}\n")
        
        # Add a delay to avoid hitting API rate limits
        time.sleep(1)

print("Processing complete. Descriptions saved to MIMIC_ALL_text.txt")