# Experiment Log

## 2022-08-22

### 64 Embedding size

```shell
$ python3 factorization_simple.py --num_epochs=25 --batch_size=1024 --learning_rate=1e-3 --embedding_dim=64
```

### 32 Embedding size

```shell
$ python3 factorization_simple.py --num_epochs=25 --batch_size=1024 --learning_rate=1e-3 --embedding_dim=32
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
