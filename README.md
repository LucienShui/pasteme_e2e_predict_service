# PasteMe End to End Predict Service

PasteMe model end to end predict service

## Deploy

Download trained model from [github.com/PasteUs/PasteMeRIM/releases](https://github.com/PasteUs/PasteMeRIM/releases).

```shell script
mkdir -p static/models/PasteMeRIM
tar -xzvf saved_model.tar.gz
mv saved_model/PasteMeRIM static/models/PasteMeRIM/1
mv saved_model/word2idx.json static
```

```shell script
docker-compose up -d
```

Go to [localhost:5000/testPage](http://localhost:5000/testPage).
