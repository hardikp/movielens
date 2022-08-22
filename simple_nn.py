import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
from sklearn.model_selection import train_test_split

BATCH_SIZE = 1024

movies_df = pd.read_csv("~/data/ml-25m/movies.csv")
ratings_df = pd.read_csv("~/data/ml-25m/ratings.csv")
movie_to_idx = {m: i for i, m in enumerate(movies_df["movieId"].unique())}
user_to_idx = {u: i for i, u in enumerate(ratings_df["userId"].unique())}
num_users = len(user_to_idx)
num_movies = len(movie_to_idx)
print("num_users:", num_users)
print("num_movies:", num_movies)
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


class MovieLensDataset(Dataset):
    def __init__(self, movies_df, ratings_df, movie_to_idx, user_to_idx):
        self.movie_title = movies_df.set_index("movieId")["title"].T.to_dict()
        self.movie_genres = movies_df.set_index("movieId")["genres"].T.to_dict()
        self.ratings_df = ratings_df
        self.movie_to_idx = movie_to_idx
        self.user_to_idx = user_to_idx

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
            "user_id": torch.tensor([user_to_idx[user_id]], dtype=torch.long),
            "movie_id": torch.tensor([movie_to_idx[movie_id]], dtype=torch.long),
            "rating": torch.tensor([rating], dtype=torch.float),
            "movie_title": movie_title,
            "movie_genres": movie_genres,
        }


train_dataset = MovieLensDataset(movies_df, ratings_train_df, movie_to_idx, user_to_idx)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

test_dataset = MovieLensDataset(movies_df, ratings_test_df, movie_to_idx, user_to_idx)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

# Model
class SimpleNN(nn.Module):
    def __init__(self, user_vocab_size, movie_vocab_size, embedding_dim):
        super(SimpleNN, self).__init__()
        self.user_embeds = nn.Embedding(user_vocab_size, embedding_dim)
        self.movie_embeds = nn.Embedding(movie_vocab_size, embedding_dim)

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


# Training
learning_rate = 1e-3
num_epochs = 5
embedding_dim = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleNN(num_users, num_movies, embedding_dim)
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
training_log = []
training_log_filepath = "training_log_{}.txt".format(
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)


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


def write_training_log(epoch, train_running_loss, train_acc, test_acc):
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


for epoch in range(num_epochs):
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

    model = model.eval()

    for i, data in enumerate(tqdm(test_dataloader)):
        user_idx = torch.squeeze(data["user_id"]).to(device)
        movie_idx = torch.squeeze(data["movie_id"]).to(device)
        labels = torch.squeeze(data["rating"]).to(device)
        logits = model(user_idx, movie_idx)

        test_corrects += get_correct_predictions(logits, labels)
        test_count += logits.shape[0]

    write_training_log(
        epoch,
        train_running_loss / train_count * BATCH_SIZE,
        train_corrects / train_count * 100.0,
        test_corrects / test_count * 100.0,
    )
