from builtins import enumerate, len, open, print, range, super
from dataclasses import dataclass
from typing import Mapping
import datetime
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

FLAGS = flags.FLAGS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

flags.DEFINE_integer("batch_size", 512, "Batch size")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
flags.DEFINE_integer("embedding_dim", 128, "Embedding dimension")
flags.DEFINE_integer("num_epochs", 5, "Num epochs")
flags.DEFINE_string("data_dir", "~/data/ml-25m", "MovieLens data directory")
flags.DEFINE_boolean("debug", False, "Debug flag")
flags.DEFINE_float("l2_regularization_factor", 0.0, "L2 regularization factor")
flags.DEFINE_boolean("learn_biases", False, "Learn user and movie biases")


@dataclass
class Data:
    """Class for storing various data objects"""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    movies_df: pd.DataFrame
    movie_map: Mapping[int, int]
    user_map: Mapping[int, int]

    def num_users(self) -> int:
        return len(self.user_map)

    def num_movies(self) -> int:
        return len(self.movie_map)


def load_data(data_dir):
    movies_df = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings_df = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movie_map = {m: i for i, m in enumerate(movies_df["movieId"].unique())}
    user_map = {u: i for i, u in enumerate(ratings_df["userId"].unique())}
    num_users = len(user_map)
    num_movies = len(movie_map)
    print("num_users:", num_users)
    print("num_movies:", num_movies)
    print("num_ratings:", ratings_df.shape[0])

    train_df, test_df = train_test_split(ratings_df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index()[["userId", "movieId", "rating", "timestamp"]]
    test_df = test_df.reset_index()[["userId", "movieId", "rating", "timestamp"]]

    data = Data(train_df, test_df, movies_df, movie_map, user_map)
    return data


class MovieLensDataset(Dataset):
    def __init__(self, data, ratings_df):
        self.movie_title = data.movies_df.set_index("movieId")["title"].T.to_dict()
        self.movie_genres = data.movies_df.set_index("movieId")["genres"].T.to_dict()
        self.ratings_df = ratings_df
        self.movie_map = data.movie_map
        self.user_map = data.user_map

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        user_id = self.ratings_df["userId"][idx]
        movie_id = self.ratings_df["movieId"][idx]
        rating = self.ratings_df["rating"][idx]
        movie_title = self.movie_title[movie_id]
        movie_genres = self.movie_genres[movie_id]

        return {
            "user_id": torch.tensor([self.user_map[user_id]], dtype=torch.long),
            "movie_id": torch.tensor([self.movie_map[movie_id]], dtype=torch.long),
            "rating": torch.tensor([rating], dtype=torch.float),
            "movie_title": movie_title,
            "movie_genres": movie_genres,
        }


# Matrix Factorization model
class Factorization(nn.Module):
    def __init__(self, num_users, num_movies):
        super(Factorization, self).__init__()
        self.user_embeds = nn.Embedding(num_users, FLAGS.embedding_dim)
        self.movie_embeds = nn.Embedding(num_movies, FLAGS.embedding_dim)

    def forward(self, user_idx, movie_idx):
        user = self.user_embeds(user_idx)
        movie = self.movie_embeds(movie_idx)
        similarity = F.cosine_similarity(user, movie)

        # User ratings can vary from 0.5 to 5.0 in increments of 0.5:
        # 0.5, 1.0, 1.5, ..., 4.5, 5.0
        # Adjust the similarity to map the output to 0.25 to 5.25.
        # This way, predicted values:
        # - between 0.25 and 0.75 can be counted towards 0.5 rating.
        # - between 0.75 and 1.25 can be counted towards 1.0 rating.
        # And so on.
        # Cosine similarity can be between -1 and 1.
        similarity = similarity * 2.5 + 2.75
        return similarity


# Matrix Factorization model with user & item biases
class FactorizationBias(nn.Module):
    def __init__(self, num_users, num_movies):
        super(FactorizationBias, self).__init__()
        self.user_embeds = nn.Embedding(num_users, FLAGS.embedding_dim)
        self.movie_embeds = nn.Embedding(num_movies, FLAGS.embedding_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)

    def forward(self, user_idx, movie_idx):
        user = self.user_embeds(user_idx)
        movie = self.movie_embeds(movie_idx)
        similarity = F.cosine_similarity(user, movie)
        # User ratings can vary from 0.5 to 5.0 in increments of 0.5:
        # 0.5, 1.0, 1.5, ..., 4.5, 5.0
        # Adjust the similarity to map the output to 0.25 to 5.25.
        # This way, predicted values:
        # - between 0.25 and 0.75 can be counted towards 0.5 rating.
        # - between 0.75 and 1.25 can be counted towards 1.0 rating.
        # And so on.
        # Cosine similarity can be between -1 and 1.
        similarity = similarity * 2.5 + 2.75

        movie_bias = self.movie_biases(movie_idx)
        user_bias = self.user_biases(user_idx)
        prediction = similarity + user_bias.squeeze() + movie_bias.squeeze()
        return prediction


def get_correct_predictions(logit, target):
    batch_size = logit.shape[0]
    diff = torch.abs(target - logit)
    corrects = torch.less(diff, torch.ones(batch_size).to(device) * 0.25).sum()
    return corrects


@dataclass
class LogData:
    """Dataclass holding training epoch log data"""

    epoch: int
    train_loss: float
    train_acc: float
    test_acc: float


def get_epoch_summary(log_data: LogData):
    return "Epoch: %d | Loss: %.4f | Train Accuracy: %.2f | Test Accuracy: %.2f" % (
        log_data.epoch,
        log_data.train_loss,
        log_data.train_acc,
        log_data.test_acc,
    )


def load_model(model):
    log_path, model_path = get_path()
    if not os.path.exists(log_path):
        print(f"{log_path} doesn't exist. Not loading model.")
        return None
    if not os.path.exists(model_path):
        print(f"{model_path} doesn't exist. Not loading model.")
        return None

    # Load model parameters
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))

    # Load max epoch
    lines = open(log_path).read().strip().split("\n")
    lines = [l for l in lines if not l.startswith("#")]
    last_line = lines[-1]
    max_epoch = int(last_line.split(" | ")[0].split(": ")[1])
    return model, max_epoch + 1


def get_path():
    prefix = sys.argv[0].split(".")[0]
    filename = f"{prefix}_{FLAGS.num_epochs}_{FLAGS.batch_size}"
    filename += f"_{FLAGS.learning_rate}_{FLAGS.embedding_dim}"
    filename += f"_{FLAGS.l2_regularization_factor}_{FLAGS.learn_biases}"
    return filename + ".txt", filename + ".model"


def checkpoint(epoch_data, model):
    log_path, model_path = get_path()

    # Read existing logs
    lines = []
    if os.path.exists(log_path):
        lines = open(log_path).read().strip().split("\n")
        lines = [l for l in lines if not l.startswith("#")]
    print(get_epoch_summary(epoch_data))
    lines.append(get_epoch_summary(epoch_data))

    # Write the model
    torch.save(model.state_dict(), model_path)

    # Write log file
    f = open(log_path, "w")
    s = ""
    for line in lines:
        s += line + "\n"
    f.write(s)
    f.close()


def main(argv):
    print("Batch size:", FLAGS.batch_size)
    print("Embedding size:", FLAGS.embedding_dim)
    print("Learning rate:", FLAGS.learning_rate)
    print("Num epochs:", FLAGS.num_epochs)
    print("L2 regularization factor:", FLAGS.l2_regularization_factor)
    print("Learn biases:", FLAGS.learn_biases)

    # Load data
    data = load_data(FLAGS.data_dir)

    # Dataloader
    train_dataset = MovieLensDataset(data, data.train_df)
    train_dataloader = DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0
    )

    test_dataset = MovieLensDataset(data, data.test_df)
    test_dataloader = DataLoader(
        test_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0
    )

    if FLAGS.learn_biases is False:
        model = Factorization(data.num_users(), data.num_movies())
    else:
        model = FactorizationBias(data.num_users(), data.num_movies())
    print(model)

    # Load model from the checkpoint if it exists
    out = load_model(model)
    if out is not None:
        (model, start_epoch) = out
    else:
        start_epoch = 0

    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FLAGS.learning_rate,
        weight_decay=FLAGS.l2_regularization_factor,
    )

    # Train + Eval
    for epoch in range(start_epoch, FLAGS.num_epochs):
        train_running_loss = 0.0
        train_corrects = 0.0
        train_count = 0
        test_corrects = 0.0
        test_count = 0
        model = model.train()

        for i, data in enumerate(tqdm(train_dataloader)):
            user_idx = torch.squeeze(data["user_id"]).to(device)
            movie_idx = torch.squeeze(data["movie_id"]).to(device)
            labels = torch.squeeze(data["rating"]).to(device)
            logits = model(user_idx, movie_idx)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_running_loss += loss.detach().item()
            train_corrects += get_correct_predictions(logits, labels)
            train_count += logits.shape[0]

            if FLAGS.debug and i == 2:
                break

        model = model.eval()

        for i, data in enumerate(tqdm(test_dataloader)):
            user_idx = torch.squeeze(data["user_id"]).to(device)
            movie_idx = torch.squeeze(data["movie_id"]).to(device)
            labels = torch.squeeze(data["rating"]).to(device)
            logits = model(user_idx, movie_idx)

            test_corrects += get_correct_predictions(logits, labels)
            test_count += logits.shape[0]

            if FLAGS.debug and i == 2:
                break

        train_loss = train_running_loss / train_count * FLAGS.batch_size
        train_acc = train_corrects / train_count * 100.0
        test_acc = test_corrects / test_count * 100.0
        epoch_data = LogData(epoch, train_loss, train_acc, test_acc)
        checkpoint(epoch_data, model)


if __name__ == "__main__":
    app.run(main)
