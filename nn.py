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

flags.DEFINE_integer("batch_size", 1024, "Batch size")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
flags.DEFINE_integer("embedding_dim", 16, "Embedding dimension")
flags.DEFINE_integer("num_epochs", 5, "Num epochs")
flags.DEFINE_string("data_dir", "~/data/ml-25m", "MovieLens data directory")
flags.DEFINE_boolean("debug", False, "Debug flag")
flags.DEFINE_float("l2_regularization_factor", 0.0, "L2 regularization factor")
flags.DEFINE_float("dropout", 0.25, "Dropout")
flags.DEFINE_boolean("apply_emb_dropout", True, "Apply Dropout to Embeddings")
flags.DEFINE_integer("l_size_user", 16, "user_id linear layer size")
flags.DEFINE_integer("l_size_movie", 16, "movie_id linear layer size")
flags.DEFINE_integer("l_size_genre", 16, "genre_ids linear layer size")
flags.DEFINE_integer("l_size_year", 16, "year linear layer size")


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

        return {
            "user_id": torch.tensor([self.user_to_idx[user_id]], dtype=torch.long),
            "movie_id": torch.tensor([self.movie_to_idx[movie_id]], dtype=torch.long),
            "rating": torch.tensor([rating], dtype=torch.float),
            "genres": [self.genre_to_idx[g] for g in genres],
            "year": torch.tensor([self.year_to_idx[year]], dtype=torch.long),
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


# Model
class NeuralNet(nn.Module):
    def __init__(
        self,
        user_vocab_size,
        movie_vocab_size,
        genre_vocab_size,
        year_vocab_size,
        embedding_dim,
        dropout_rate,
    ):
        super(NeuralNet, self).__init__()
        self.user_embeds = nn.Embedding(user_vocab_size, embedding_dim)
        self.user_linear = nn.Linear(embedding_dim, FLAGS.l_size_user)

        self.movie_embeds = nn.Embedding(movie_vocab_size, embedding_dim)
        self.movie_linear = nn.Linear(embedding_dim, FLAGS.l_size_movie)

        if FLAGS.l_size_genre > 0:
            self.genre_embeds = nn.EmbeddingBag(genre_vocab_size, embedding_dim)
            self.genre_linear = nn.Linear(embedding_dim, FLAGS.l_size_genre)

        if FLAGS.l_size_year > 0:
            self.year_embeds = nn.Embedding(year_vocab_size, embedding_dim)
            self.year_linear = nn.Linear(embedding_dim, FLAGS.l_size_year)

        in_size = (
            FLAGS.l_size_user
            + FLAGS.l_size_movie
            + FLAGS.l_size_genre
            + FLAGS.l_size_year
            + 1
        )
        self.combined_linear = nn.Linear(in_size, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, user_idx, movie_idx, genre_idxs, genre_offsets, year_idx):
        user = self.user_embeds(user_idx)
        movie = self.movie_embeds(movie_idx)

        if FLAGS.apply_emb_dropout:
            user = self.dropout(user)
            movie = self.dropout(movie)

        similarity = F.cosine_similarity(user, movie)
        similarity = self.dropout(similarity)

        user = self.dropout(F.relu(self.user_linear(user)))
        movie = self.dropout(F.relu(self.movie_linear(movie)))

        if FLAGS.l_size_genre > 0:
            genres = self.genre_embeds(genre_idxs, genre_offsets)
            if FLAGS.apply_emb_dropout:
                genres = self.dropout(genres)
            genres = self.dropout(F.relu(self.genre_linear(genres)))

        if FLAGS.l_size_year > 0:
            year = self.year_embeds(year_idx)
            if FLAGS.apply_emb_dropout:
                year = self.dropout(year)
            year = self.dropout(F.relu(self.year_linear(year)))

        similarity = torch.reshape(similarity, (-1, 1))
        if FLAGS.l_size_genre > 0 and FLAGS.l_size_year > 0:
            out = torch.cat([user, movie, similarity, genres, year], 1)
        elif FLAGS.l_size_genre > 0:
            out = torch.cat([user, movie, similarity, genres], 1)
        elif FLAGS.l_size_year > 0:
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


def get_file_name():
    prefix = sys.argv[0].split(".")[0]
    filename = f"{prefix}_{FLAGS.num_epochs}_{FLAGS.batch_size}"
    filename += f"_{FLAGS.learning_rate}_{FLAGS.embedding_dim}"
    filename += f"_{FLAGS.l2_regularization_factor}_{FLAGS.dropout}"
    filename += f"_{FLAGS.l_size_user}_{FLAGS.l_size_movie}"
    filename += f"_{FLAGS.l_size_genre}_{FLAGS.l_size_year}"
    filename += f'_{datetime.datetime.now().strftime("%m%d%H%M%S")}.txt'
    return filename


def main(argv):
    print("Batch size:", FLAGS.batch_size)
    print("Embedding size:", FLAGS.embedding_dim)
    print("Learning rate:", FLAGS.learning_rate)
    print("Num epochs:", FLAGS.num_epochs)
    print("L2 regularization factor:", FLAGS.l2_regularization_factor)
    print("Dropout:", FLAGS.dropout)
    print("l_size_user:", FLAGS.l_size_user)
    print("l_size_movie:", FLAGS.l_size_movie)
    print("l_size_genre:", FLAGS.l_size_genre)
    print("l_size_year:", FLAGS.l_size_year)

    # Load data
    (
        movies_df,
        ratings_train_df,
        ratings_test_df,
        movie_to_idx,
        user_to_idx,
        genre_to_idx,
        year_to_idx,
    ) = load_data(FLAGS.data_dir)

    # Dataloader
    train_dataset = MovieLensDataset(
        movies_df,
        ratings_train_df,
        movie_to_idx,
        user_to_idx,
        genre_to_idx,
        year_to_idx,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    test_dataset = MovieLensDataset(
        movies_df, ratings_test_df, movie_to_idx, user_to_idx, genre_to_idx, year_to_idx
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = NeuralNet(
        len(user_to_idx),
        len(movie_to_idx),
        len(genre_to_idx),
        len(year_to_idx),
        FLAGS.embedding_dim,
        FLAGS.dropout,
    )
    model = model.to(device)
    print(model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FLAGS.learning_rate,
        weight_decay=FLAGS.l2_regularization_factor,
    )
    training_log = []
    training_log_filepath = get_file_name()

    # Train + Eval
    for epoch in range(FLAGS.num_epochs):
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
