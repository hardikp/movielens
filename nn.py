from dataclasses import dataclass
import datetime
import os
import sys
from typing import Mapping

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

flags.DEFINE_integer("batch_size", 1024, "Batch size")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
flags.DEFINE_integer("embedding_dim", 16, "Embedding dimension")
flags.DEFINE_integer("num_epochs", 5, "Num epochs")
flags.DEFINE_string("data_dir", "~/data/ml-25m", "MovieLens data directory")
flags.DEFINE_boolean("debug", False, "Debug flag")
flags.DEFINE_float("l2", 0.0, "L2 regularization factor")
flags.DEFINE_float("dropout", 0.25, "Dropout")
flags.DEFINE_string("l_size_user", "16", "user_id linear layer sizes comma separated")
flags.DEFINE_string("l_size_movie", "16", "movie_id linear layer sizes comma separated")
flags.DEFINE_string(
    "l_size_genre", "16", "genre_ids linear layer sizes comma separated"
)
flags.DEFINE_string("l_size_year", "16", "year linear layer sizes comma separated")
flags.DEFINE_string("norm_type", "LayerNorm", "Specify Batch/Layer Normalization")
flags.DEFINE_boolean(
    "init_weights", True, "Initialize weights using xavier initialization"
)


def get_year(title):
    try:
        return int(title.split("(")[-1].split(")")[0])
    except:
        return 0


@dataclass
class Data:
    """Class for storing various data objects"""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    movies_df: pd.DataFrame
    movie_map: Mapping[int, int]
    user_map: Mapping[int, int]
    genre_map: Mapping[int, int]
    year_map: Mapping[int, int]

    def num_users(self) -> int:
        return len(self.user_map)

    def num_movies(self) -> int:
        return len(self.movie_map)

    def num_genres(self) -> int:
        return len(self.genre_map)

    def num_years(self) -> int:
        return len(self.year_map)


def load_data(data_dir):
    movies_df = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings_df = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movie_map = {m: i for i, m in enumerate(movies_df["movieId"].unique())}
    user_map = {u: i for i, u in enumerate(ratings_df["userId"].unique())}
    genres = movies_df["genres"].apply(lambda g: g.split("|"))
    genres_unique = genres.apply(pd.Series).stack().reset_index(drop=True).unique()
    genre_map = {u: i for i, u in enumerate(genres_unique)}
    year = movies_df["title"].apply(lambda t: get_year(t))
    year_map = {u: i for i, u in enumerate(year.unique())}
    num_users = len(user_map)
    num_movies = len(movie_map)
    num_genres = len(genre_map)
    num_years = len(year_map)
    print("num_users:", num_users)
    print("num_movies:", num_movies)
    print("num_genres:", num_genres)
    print("num_years:", num_years)
    print("num_ratings:", ratings_df.shape[0])

    train_df, test_df = train_test_split(ratings_df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index()[["userId", "movieId", "rating", "timestamp"]]
    test_df = test_df.reset_index()[["userId", "movieId", "rating", "timestamp"]]

    data = Data(train_df, test_df, movies_df, movie_map, user_map, genre_map, year_map)
    return data


class MovieLensDataset(Dataset):
    def __init__(self, data, ratings_df):
        self.movie_title = data.movies_df.set_index("movieId")["title"].T.to_dict()
        self.movie_genres = data.movies_df.set_index("movieId")["genres"].T.to_dict()
        self.ratings_df = ratings_df
        self.movie_map = data.movie_map
        self.user_map = data.user_map
        self.genre_map = data.genre_map
        self.year_map = data.year_map

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        user_id = self.ratings_df["userId"][idx]
        movie_id = self.ratings_df["movieId"][idx]
        rating = self.ratings_df["rating"][idx]
        movie_title = self.movie_title[movie_id]
        year = get_year(movie_title)
        movie_genres = self.movie_genres[movie_id]
        genres = movie_genres.split("|")

        return {
            "user_id": torch.tensor([self.user_map[user_id]], dtype=torch.long),
            "movie_id": torch.tensor([self.movie_map[movie_id]], dtype=torch.long),
            "rating": torch.tensor([rating], dtype=torch.float),
            "genres": [self.genre_map[g] for g in genres],
            "year": torch.tensor([self.year_map[year]], dtype=torch.long),
        }


def collate_fn(batch):
    user_ids = torch.stack([d["user_id"] for d in batch]).to(device)
    user_ids = torch.squeeze(user_ids)
    movie_ids = torch.stack([d["movie_id"] for d in batch]).to(device)
    movie_ids = torch.squeeze(movie_ids)
    ratings = torch.stack([d["rating"] for d in batch]).to(device)
    ratings = torch.squeeze(ratings)
    years = torch.stack([d["year"] for d in batch]).to(device)
    years = torch.squeeze(years)
    genres = []
    genre_offsets = []
    current_offset = 0
    for d in batch:
        genres.extend(d["genres"])
        genre_offsets.append(current_offset)
        current_offset += len(d["genres"])
    genres = torch.tensor(genres, dtype=torch.long).to(device)
    genre_offsets = torch.tensor(genre_offsets, dtype=torch.long).to(device)
    return (user_ids, movie_ids, ratings, genres, genre_offsets, years)


# Function to create a user/movie/genre/year tower
def get_tower(l_size):
    layer_sizes = [int(i) for i in l_size.split(",")]
    layers = []
    in_size = FLAGS.embedding_dim
    for user_layer_size in layer_sizes:
        # Linear layer
        l = nn.Linear(in_size, user_layer_size)
        if FLAGS.init_weights:
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_uniform_(l.weight, gain=gain)
        layers.append(l)

        # Optionally, initialize the weights
        if FLAGS.norm_type == "LayerNorm":
            layers.append(nn.LayerNorm(user_layer_size))
        elif FLAGS.norm_type == "BatchNorm":
            layers.append(nn.BatchNorm1d(user_layer_size))

        # Activation function
        layers.append(nn.ReLU())
        # Dropout
        layers.append(nn.Dropout(FLAGS.dropout))
        in_size = user_layer_size

    tower = nn.Sequential(*layers)
    return tower, layer_sizes[-1]


# Function to create embedding layers
def get_embeddings(count, create_bag=False):
    if create_bag:
        embeds = nn.EmbeddingBag(count, FLAGS.embedding_dim)
    else:
        embeds = nn.Embedding(count, FLAGS.embedding_dim)

    # Optionally, initialize the weights
    if FLAGS.init_weights:
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(embeds.weight, gain=gain)

    return embeds


# Model
class NeuralNet(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, num_years):
        super(NeuralNet, self).__init__()
        self.user_embeds = get_embeddings(num_users)
        self.user_tower, user_size = get_tower(FLAGS.l_size_user)
        self.movie_embeds = get_embeddings(num_movies)
        self.movie_tower, movie_size = get_tower(FLAGS.l_size_movie)
        if len(FLAGS.l_size_genre) > 0:
            self.genre_embeds = get_embeddings(num_genres, create_bag=True)
            self.genre_tower, genre_size = get_tower(FLAGS.l_size_genre)
        else:
            genre_size = 0
        if len(FLAGS.l_size_year) > 0:
            self.year_embeds = get_embeddings(num_years)
            self.year_tower, year_size = get_tower(FLAGS.l_size_year)
        else:
            year_size = 0

        in_size = user_size + movie_size + genre_size + year_size + 1
        self.combined_linear = nn.Linear(in_size, 1)
        self.dropout = nn.Dropout(FLAGS.dropout)

    def forward(self, user_idx, movie_idx, genre_idxs, genre_offsets, year_idx):
        user = self.user_embeds(user_idx)
        movie = self.movie_embeds(movie_idx)
        similarity = F.cosine_similarity(user, movie)
        similarity = self.dropout(similarity)

        user = self.user_tower(user)
        movie = self.movie_tower(movie)

        if len(FLAGS.l_size_genre) > 0:
            genres = self.genre_embeds(genre_idxs, genre_offsets)
            genres = self.genre_tower(genres)

        if len(FLAGS.l_size_year) > 0:
            year = self.year_embeds(year_idx)
            year = self.year_tower(year)

        similarity = torch.reshape(similarity, (-1, 1))
        if len(FLAGS.l_size_genre) > 0 and len(FLAGS.l_size_year) > 0:
            out = torch.cat([user, movie, similarity, genres, year], 1)
        elif len(FLAGS.l_size_genre) > 0:
            out = torch.cat([user, movie, similarity, genres], 1)
        elif len(FLAGS.l_size_year) > 0:
            out = torch.cat([user, movie, similarity, year], 1)
        else:
            out = torch.cat([user, movie, similarity], 1)
        out = torch.sigmoid(self.combined_linear(out))
        out = torch.reshape(out, (-1,))

        # User ratings can vary from 0.5 to 5.0 in increments of 0.5:
        # 0.5, 1.0, 1.5, ..., 4.5, 5.0
        # Map the output to 0.25 to 5.25.
        # This way, predicted values:
        # - between 0.25 and 0.75 can be counted towards 0.5 rating.
        # - between 0.75 and 1.25 can be counted towards 1.0 rating.
        # And so on.
        # Sigmoid output can be between 0 and 1.
        out = out * 5.0 + 0.25
        return out


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
    filename += f"_{FLAGS.l2}_{FLAGS.dropout}"
    filename += f"_{FLAGS.l_size_user}_{FLAGS.l_size_movie}"
    filename += f"_{FLAGS.l_size_genre}_{FLAGS.l_size_year}"
    filename += f"_{FLAGS.norm_type}_{FLAGS.init_weights}"
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
    print("L2 regularization factor:", FLAGS.l2)
    print("Dropout:", FLAGS.dropout)
    print("l_size_user:", FLAGS.l_size_user)
    print("l_size_movie:", FLAGS.l_size_movie)
    print("l_size_genre:", FLAGS.l_size_genre)
    print("l_size_year:", FLAGS.l_size_year)
    print("norm_type:", FLAGS.norm_type)
    print("init_weights:", FLAGS.init_weights)

    # Load data
    data = load_data(FLAGS.data_dir)

    # Dataloader
    train_dataset = MovieLensDataset(data, data.train_df)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    test_dataset = MovieLensDataset(data, data.test_df)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = NeuralNet(
        data.num_users(), data.num_movies(), data.num_genres(), data.num_years()
    )
    model = model.to(device)
    print(model)

    # Load model from the checkpoint if it exists
    out = load_model(model)
    if out is not None:
        (model, start_epoch) = out
    else:
        start_epoch = 0

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2
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
            user_idx, movie_idx, labels, genre_idxs, genre_offsets, year_idx = data
            logits = model(user_idx, movie_idx, genre_idxs, genre_offsets, year_idx)
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
            user_idx, movie_idx, labels, genre_idxs, genre_offsets, year_idx = data
            logits = model(user_idx, movie_idx, genre_idxs, genre_offsets, year_idx)

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
