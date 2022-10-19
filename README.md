# movielens

## Matrix Factorization

Matrix Factorization without user/movie biases:
```shell
python3 mf.py --num_epochs=50 \
              --batch_size=512 \
              --learning_rate=1e-3 \
              --embedding_dim=128 \
              --l2_regularization_factor=1e-6 \
              --learn_biases=False
```

Matrix Factorization with user/movie biases:
```shell
python3 mf.py --num_epochs=50 \
              --batch_size=512 \
              --learning_rate=1e-3 \
              --embedding_dim=128 \
              --l2_regularization_factor=1e-6 \
              --learn_biases=True
```

Matrix Factorization with user/movie biases + side features:
```shell
python3 mf_side_features.py --num_epochs=50 \
                            --batch_size=512 \
                            --learning_rate=1e-3 \
                            --embedding_dim=128 \
                            --l2_regularization_factor=1e-6 \
                            --learn_biases=True
```
