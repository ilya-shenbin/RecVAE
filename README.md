# RecVAE
The official PyTorch implementation of the paper "RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback".

In order to train RecVAE on [MovieLens-20M](http://files.grouplens.org/datasets/movielens/ml-20m.zip) dataset, preprocess it using following script:

```sh
python preprocessing.py --dataset <path_to_csv_file> --output_dir ml-20m/pro_sg/ --threshold 3.5 --heldout_users 10000
```

You can also use another dataset, it should contain columns `userId`, `movieId` and `rating` (in arbitrary order).

Finally, run all cells in `recvae.ipynb`.

Some sources from  [Variational autoencoders for collaborative filtering](https://github.com/dawenl/vae_cf) is partially used.

If you used this code for a publication, please cite our WSDM'20 paper
```
@inproceedings{10.1145/3336191.3371831,
  author = {Shenbin, Ilya and Alekseev, Anton and Tutubalina, Elena and Malykh, Valentin and Nikolenko, Sergey I.},
  title = {RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback},
  year = {2020},
  isbn = {9781450368223},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3336191.3371831},
  doi = {10.1145/3336191.3371831},
  booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages = {528–536},
  numpages = {9},
  keywords = {deep learning, collaborative filtering, variational autoencoders},
  location = {Houston, TX, USA},
  series = {WSDM ’20}
}
```

