import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import os


def compute_and_save_text_embeddings(
        dataset_name,
        model_name='bert-base-uncased',
        batch_size=8,
        output_dir="embeddings",
        target_name="label",
):
    """
    Pre-computes and saves BERT embeddings for a given dataset.

    Args:
    - dataset_name (str): Name of the dataset to load from the Hugging Face datasets library.
    - model_name (str): Name of the pre-trained BERT model to use.
    - batch_size (int): Batch size for processing data.
    - output_dir (str): Directory to save the embeddings.

    Returns:
    None
    """

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define a function to process the text and obtain embeddings
    def get_bert_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
        return embeddings.cpu()

    # Precompute and save embeddings
    os.makedirs(output_dir, exist_ok=True)

    for split in dataset.keys():
        data_loader = DataLoader(dataset[split], batch_size=batch_size)
        embeddings_list = []
        labels_list = []
        for batch in tqdm(data_loader, desc=f"Processing {split} split"):
            texts = batch['text']
            labels = batch[target_name]
            embeddings = get_bert_embeddings(texts)
            embeddings_list.append(embeddings)
            labels_list.extend(labels)

        embeddings_array = torch.concatenate(embeddings_list, dim=0)
        labels_array = torch.tensor(labels_list).long()
        torch.save(embeddings_array, os.path.join(output_dir, f"{split}_x.pt"))
        torch.save(labels_array, os.path.join(output_dir, f"{split}_y.pt"))