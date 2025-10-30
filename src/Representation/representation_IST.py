from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import re


def load_patient_descriptions(file_path):
    """Load and parse patient descriptions from text file."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split by delimiter and parse each patient entry
    patients = content.split('==================================================')
    descriptions = []
    
    for patient in patients:
        if not patient.strip():  # Skip empty entries
            continue
        # Extract patient ID and description
        patient_id = re.search(r'Patient ID: (\d+)', patient)
        if patient_id:
            patient_id = int(patient_id.group(1))
            description = patient.split('\n', 2)[1].strip()  # Get text after ID
            print(f"Patient ID: {patient_id}, Description: {description}")
            descriptions.append({'patient_id': patient_id, 'description': description})
    
    return pd.DataFrame(descriptions)


def get_bert_embeddings(texts, model_name='bert-base-uncased'):
    """Generate BERT embeddings for a list of texts."""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize and encode text
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = model(**inputs)
            
            # Use [CLS] token embedding as sentence representation
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
    
    return torch.tensor(embeddings)

def main():
    # Load patient descriptions
    file_path = 'dataset/IST/IST_COMPLETE.txt'
    df = load_patient_descriptions(file_path)
    
    # Get BERT embeddings
    embeddings = get_bert_embeddings(df['description'].tolist())
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    # Each row in embeddings is a 768-dimensional vector representing a patient description
    
    return embeddings, df

if __name__ == "__main__":
    embeddings, df = main()
    # Convert tensor to numpy array, then to pandas DataFrame before saving
    embeddings_df = pd.DataFrame(embeddings.numpy())
    embeddings_df.to_csv('dataset/IST/IST_ALL_embeddings.csv', index=True)
