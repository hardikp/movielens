# Experiment Log

## 2022-09-23

```shell
time python3 nn_data1.py --num_epochs=25 --batch_size=512 --learning_rate=1e-3 --embedding_dim=128 --l_size_user=16 --l_size_movie=16 --l_size_genre=4 --l_size_year=4 --dropout=0.25
```

## 2022-08-28

### Deep Neural Network

```shell
$ time python3 neural_net_ff.py --num_epochs=25 --batch_size=512 --learning_rate=1e-3 --embedding_dim=128
```

### Simple Neural Network

```shell
$ time python3 neural_net_simple.py --num_epochs=25 --batch_size=512 --learning_rate=1e-3 --embedding_dim=128
```

## 2022-08-23

### 1e-2/1e-4 learning_rate

```shell
$ python3 factorization_simple.py --num_epochs=25 --batch_size=1024 --learning_rate=1e-2 --embedding_dim=16
$ python3 factorization_simple.py --num_epochs=25 --batch_size=1024 --learning_rate=1e-4 --embedding_dim=16
```

### 512/2048 batch_sizes

```shell
$ python3 factorization_simple.py --num_epochs=25 --batch_size=512 --learning_rate=1e-3 --embedding_dim=16
$ python3 factorization_simple.py --num_epochs=25 --batch_size=2048 --learning_rate=1e-3 --embedding_dim=16
```

## 2022-08-22

### 32/64/128 Embedding size

```shell
$ python3 factorization_simple.py --num_epochs=25 --batch_size=1024 --learning_rate=1e-3 --embedding_dim=32
$ python3 factorization_simple.py --num_epochs=25 --batch_size=1024 --learning_rate=1e-3 --embedding_dim=64
$ python3 factorization_simple.py --num_epochs=25 --batch_size=1024 --learning_rate=1e-3 --embedding_dim=128
```

### 25 epochs

```shell
$ python3 factorization_simple.py --num_epochs=25 --batch_size=1024 --learning_rate=1e-3 --embedding_dim=16
```

file: factorization_simple.py
Parameters:
* learning_rate: 1e-3 (Adam optimizer)
* batch_size: 1024
* num_epochs: 25
* embedding_dim: 16
* regression loss: MSELoss (cosine similarity output was mapped to 10 classes ranging from 0.5 to 5.0)
* train / test split: 90/10
* Training time: 318m31.743s (~11m45s per epoch)

Output: training_log_20220822_071024.txt

## 2022-08-21

```shell
$ python3 factorization_simple.py --num_epochs=5 --batch_size=1024 --learning_rate=1e-3 --embedding_dim=16
```

file: factorization_simple.py
Parameters:
* learning_rate: 1e-3 (Adam optimizer)
* batch_size: 1024
* num_epochs: 5
* embedding_dim: 16
* regression loss: MSELoss (cosine similarity output was mapped to 10 classes ranging from 0.5 to 5.0)
* train / test split: 90/10
* Training time: 62m17.168s (~11m45s per epoch)

Output: training_log_20220821_210626.txt
