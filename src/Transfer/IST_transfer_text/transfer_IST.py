import pandas as pd
import time


# Function to get text description for a patient
def get_patient_description(patient_values):
    try:
        completion = model.completions.create(
            messages=[
                {"role": "user", 
                 "content": "Here is a patient current information (if this patient can use diagnostic equipment or has access to healthcare facilities):\n"
                           "AGE,SEX,RSBP,RATRIAL,RVISINF,RHEP24,RASP3,RDEF1,RDEF2,RDEF3,RDEF4,RDEF5,RDEF6,RDEF7,RDEF8.\n"
                           f"The values are {','.join(map(str, patient_values))}.\n"
                           "AGE denotes the patient's age in years, while SEX specifies the biological sex (Male or Female).\n"
                           "RATRIAL indicates the presence or absence of atrial fibrillation, a heart rhythm disorder.\n"
                           "RSBP records the patient's systolic blood pressure (in mmHg) measured at the time of randomization.\n"
                           "RVISINF shows whether an infarct (a stroke-related tissue damage) is visible on a CT scan, and\n"
                           "RHEP24 notes if the patient received heparin treatment within 24 hours prior to randomization.\n"
                           "RASP3 reflects aspirin intake within the three days leading up to randomization.\n"
                           "Lastly, RDEF1-8 captures various physical or neurological deficits, such as impairments in facial,\n"
                           "arm, or leg function, providing a comprehensive snapshot.\n"
                           "How would this patient describe his/her symptoms via text and describe his/her feeling without any diagnostic measurements? Only output the description text."}
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
with open(PATH + 'IST_ALL_text.txt', 'w', encoding='utf-8') as f:
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

print("Processing complete. Descriptions saved to IST_ALL_text.txt")