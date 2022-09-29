from builtins import enumerate, int, len, open, print, range, super
import datetime
import os

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


def get_year(title):
    try:
        return int(title.split("(")[-1].split(")")[0])
    except:
        return 0


def load_data(data_dir):
    movies_df = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings_df = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movie_to_idx = {m: i for i, m in enumerate(movies_df["movieId"].unique())}
    user_to_idx = {u: i for i, u in enumerate(ratings_df["userId"].unique())}
    genres = movies_df["genres"].apply(lambda g: g.split("|"))
    genres_unique = genres.apply(pd.Series).stack().reset_index(drop=True).unique()
    genre_to_idx = {u: i for i, u in enumerate(genres_unique)}
    year = movies_df["title"].apply(lambda t: get_year(t))
    year_to_idx = {u: i for i, u in enumerate(year.unique())}
    num_users = len(user_to_idx)
    num_movies = len(movie_to_idx)
    num_genres = len(genre_to_idx)
    num_years = len(year_to_idx)
    print("num_users:", num_users)
    print("num_movies:", num_movies)
    print("num_genres:", num_genres)
    print("num_years:", num_years)
    print("num_ratings:", ratings_df.shape[0])

    ratings_train_df, ratings_test_df = train_test_split(
        ratings_df, test_size=0.1, random_state=42
    )
    ratings_train_df = ratings_train_df.reset_index()[
        ["userId", "movieId", "rating", "timestamp"]
    ]
    ratings_test_df = ratings_test_df.reset_index()[
        ["userId", "movieId", "rating", "timestamp"]
    ]

    return (
        movies_df,
        ratings_train_df,
        ratings_test_df,
        movie_to_idx,
        user_to_idx,
        genre_to_idx,
        year_to_idx,
    )


class MovieLensDataset(Dataset):
    def __init__(
        self,
        movies_df,
        ratings_df,
        movie_to_idx,
        user_to_idx,
        genre_to_idx,
        year_to_idx,
    ):
        self.movie_title = movies_df.set_index("movieId")["title"].T.to_dict()
        self.movie_genres = movies_df.set_index("movieId")["genres"].T.to_dict()
        self.ratings_df = ratings_df
        self.movie_to_idx = movie_to_idx
        self.user_to_idx = user_to_idx
        self.genre_to_idx = genre_to_idx
        self.year_to_idx = year_to_idx

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
        # Only use the first genre
        genre = genres[0]

        return {
            "user_id": torch.tensor([self.user_to_idx[user_id]], dtype=torch.long),
            "movie_id": torch.tensor([self.movie_to_idx[movie_id]], dtype=torch.long),
            "rating": torch.tensor([rating], dtype=torch.float),
            "genre_id": torch.tensor([self.genre_to_idx[genre]], dtype=torch.long),
            "year_id": torch.tensor([self.year_to_idx[year]], dtype=torch.long),
        }


# Matrix Factorization model with user & item biases
class MFSideFeaturesBias(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, num_years, embedding_dim):
        super(MFSideFeaturesBias, self).__init__()
        self.user_embeds = nn.Embedding(num_users, embedding_dim)
        self.movie_embeds = nn.Embedding(num_movies, embedding_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)
        self.genre_embeds = nn.Embedding(num_genres, embedding_dim)
        self.year_embeds = nn.Embedding(num_years, embedding_dim)

    def forward(self, user_idx, movie_idx, genre_idx, year_idx):
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

        movie_bias = self.movie_biases(movie_idx).squeeze()
        user_bias = self.user_biases(user_idx).squeeze()
        genre = self.genre_embeds(genre_idx)
        year = self.year_embeds(year_idx)

        prediction = (
            similarity
            + user_bias
            + movie_bias
            + F.cosine_similarity(user, genre)
            + F.cosine_similarity(user, year)
        )
        return prediction


def get_correct_predictions(logit, target):
    batch_size = logit.shape[0]
    diff = torch.abs(target - logit)
    corrects = torch.less(diff, torch.ones(batch_size).to(device) * 0.25).sum()
    return corrects


def get_epoch_summary(epoch, train_running_loss, train_acc, test_acc):
    return "Epoch: %d | Loss: %.4f | Train Accuracy: %.2f | Test Accuracy: %.2f" % (
        epoch,
        train_running_loss,
        train_acc,
        test_acc,
    )


def write_training_log(
    training_log, training_log_filepath, epoch, train_running_loss, train_acc, test_acc
):
    epoch_summary = get_epoch_summary(epoch, train_running_loss, train_acc, test_acc)
    print(epoch_summary)

    training_log.append((epoch, train_running_loss, train_acc, test_acc))
    f = open(training_log_filepath, "w")
    s = ""
    for (epoch, train_running_loss, train_acc, test_acc) in training_log:
        s += get_epoch_summary(epoch, train_running_loss, train_acc, test_acc)
        s += "\n"
    f.write(s)
    f.close()


def main(argv):
    print("Batch size:", FLAGS.batch_size)
    print("Embedding size:", FLAGS.embedding_dim)
    print("Learning rate:", FLAGS.learning_rate)
    print("Num epochs:", FLAGS.num_epochs)
    print("L2 regularization factor:", FLAGS.l2_regularization_factor)

    # Load data
    data = load_data(FLAGS.data_dir)
    movies, train_df, test_df, movie_map, user_map, genre_map, year_map = data

    # Dataloader
    train_dataset = MovieLensDataset(
        movies, train_df, movie_map, user_map, genre_map, year_map
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0
    )

    test_dataset = MovieLensDataset(
        movies, test_df, movie_map, user_map, genre_map, year_map
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0
    )

    model = MFSideFeaturesBias(
        len(user_map),
        len(movie_map),
        len(genre_map),
        len(year_map),
        FLAGS.embedding_dim,
    )
    print(model)
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FLAGS.learning_rate,
        weight_decay=FLAGS.l2_regularization_factor,
    )
    training_log = []
    training_log_filepath = "training_log_{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # Train + Eval
    for epoch in range(FLAGS.num_epochs):
        train_running_loss = 0.0
        train_corrects = 0.0
        train_count = 0
        test_corrects = 0.0
        test_count = 0
        model = model.train()

        for i, data in enumerate(tqdm(train_dataloader)):
            user_idx = torch.squeeze(data["user_id"]).to(device)
            movie_idx = torch.squeeze(data["movie_id"]).to(device)
            genre_idx = torch.squeeze(data["genre_id"]).to(device)
            year_idx = torch.squeeze(data["year_id"]).to(device)
            labels = torch.squeeze(data["rating"]).to(device)
            logits = model(user_idx, movie_idx, genre_idx, year_idx)
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
            genre_idx = torch.squeeze(data["genre_id"]).to(device)
            year_idx = torch.squeeze(data["year_id"]).to(device)
            labels = torch.squeeze(data["rating"]).to(device)
            logits = model(user_idx, movie_idx, genre_idx, year_idx)

            test_corrects += get_correct_predictions(logits, labels)
            test_count += logits.shape[0]

            if FLAGS.debug and i == 2:
                break

        write_training_log(
            training_log,
            training_log_filepath,
            epoch,
            train_running_loss / train_count * FLAGS.batch_size,
            train_corrects / train_count * 100.0,
            test_corrects / test_count * 100.0,
        )


if __name__ == "__main__":
    app.run(main)
