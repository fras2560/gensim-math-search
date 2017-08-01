# gensim-tutorial
Going through a gensim tutorial

## Help
For all the following commands for more help try
```
python XXX.py -h
```


### To create the Models
```
python create_models.py -lda -lsi -tfidf -hdp ~/GT/tutorialDocuments ~/GT/testModel
```

### To create a Index
```
python create_index.py ~/GT/tutorialDocuments ~ /GT/testIndex/ ~/GT/testModel/model.lda 2 LDA
```

### To run a query
```
python query.py -tfidf -lsi ~/GT/data/indexes/ ntcir ~/GT/data/models/ ~/resources/documents/ ~/GT/data/results/ ~/resources/query/NTCIR12-Math-queries-participants.xml ~/resources/results/NTCIR12_Math-qrels_agg.dat
```

