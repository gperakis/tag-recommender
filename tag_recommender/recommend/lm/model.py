# Description: This script fine-tunes a pre-trained sentence-transformer
# model on a dataset of triplets of tags. The fine-tuned model can be used
# to recommend tags for a given set of input tags. The script also includes
# a function to preprocess the dataset and create triplets of tags for training.

import logging

import numpy as np
import pandas as pd

# the following import is necessary for the code to work
from datasets import Dataset  # noqa
from sentence_transformers import SentenceTransformer, losses
from torch import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from tag_recommender.recommend.lm.dataset import TripletDataset
from tag_recommender.recommend.lm.utils import (
    collate_fn,
    create_triplets_dataset,
    get_device,
    preprocess_tags,
)

tqdm.pandas()

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def preprocess_dataset(
    raw_dataset_file_path: str = "data/full_dataset.csv", sample_size: int = 100_000
) -> str:
    """
    Preprocess the dataset and create triplets of tags for training.

    Parameters
    ----------
    raw_dataset_file_path : str (default='data/full_dataset.csv')
        File path to the raw dataset.
    sample_size : int
        Number of samples to use from the dataset. Default is 100,000.

    Returns
    -------
    str
        File path to the triplet dataset.
    """
    # Load the dataset
    logger.info("Loading dataset...")
    df = pd.read_csv(raw_dataset_file_path).sample(sample_size)
    df["tags"] = df["tags"].fillna("")
    df["root_tags"] = df["root_tags"].fillna("")

    logger.info("Preprocessing tags...")
    # Apply preprocessing to dataset
    df["all_tags"] = df.progress_apply(preprocess_tags, axis=1)
    logger.info("Preprocessing complete.")
    logger.info("Creating Triplet Training Data...")
    # Flatten the list of pairs across all posts for training
    all_tags = df["all_tags"].explode().dropna().unique()

    file_path = create_triplets_dataset(df["all_tags"], all_tags, file_path=None)

    return file_path


def train_triplet_loss_tag_model(
    raw_dataset_file_path: str = "data/full_dataset.csv",
    sample_size: int = 100_000,
    epochs: int = 1,
    model_save_path: str = "artifacts/models/fine_tuned_model",
):
    # Preprocess the dataset
    triplets_file_path = preprocess_dataset(raw_dataset_file_path, sample_size)

    # Create dataset and dataloader
    logger.info("Creating DataLoader...")
    triplet_dataset = TripletDataset(triplets_file_path)
    train_dataloader = DataLoader(
        triplet_dataset,
        shuffle=True,
        batch_size=8,
        collate_fn=collate_fn,
        # num_workers > 0 raises
        # AttributeError: Can't pickle local object 'FitMixin.fit.<locals>.identity'
        num_workers=0,
    )
    logger.info("DataLoader created successfully.")

    # Initialize pre-trained sentence-transformer model
    logger.info("Loading pre-trained model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device=get_device())
    logger.info("Model loaded successfully.")

    # Define loss function (Cosine Similarity Loss for contrastive learning)
    train_loss = losses.TripletLoss(
        model,
    )
    logger.info("Fine-tuning model...")
    # Fine-tune the model with the DataLoader
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        # steps_per_epoch=500,
        epochs=epochs,
        warmup_steps=100,  # Number of warmup steps for the learning rate
        save_best_model=True,
        output_path=model_save_path,
        show_progress_bar=True,
        checkpoint_save_steps=1000,
        checkpoint_save_total_limit=2,
    )

    # Save the trained model
    logger.info("Model fine-tuned successfully.")
    logger.info(f"Saving model to {model_save_path}")
    model.save(model_save_path)
    logger.info("Model saved successfully.")


def recommend_tags(input_tags: list[str], model, all_tag_embeddings, top_k=60):
    # Get embeddings for the input tags
    input_embeddings = model.encode(input_tags)

    # Compute cosine similarity between input tags and all known tags
    # This is pretty dummy. All the tags can be in a FAISS index
    # or an ES index with cosine similarity scoring. This is just for
    # the sake of the example.
    similarities = cosine_similarity(input_embeddings, all_tag_embeddings)

    # Rank tags by similarity and return the top K recommendations
    recommended_indices = np.argsort(similarities[0])[::-1][:top_k]
    return recommended_indices
