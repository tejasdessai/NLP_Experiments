*******************************************NLP Experiments*************************************************

I am creating a classification engine that would read a set of reviews left on the webpage of a resturant and predict whether the reviews are positive or negative. The dataset has been divided into a training and test set to build and test the model.

Loading the text mining package and loading data

``` r
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)
#install.packages('tm') #text mining package
library(tm)
```

    ## Loading required package: NLP

``` r
head(dataset_original, 3)
```

    ##                                      Review Liked
    ## 1                  Wow... Loved this place.     1
    ## 2                        Crust is not good.     0
    ## 3 Not tasty and the texture was just nasty.     0

Creating a corpus to store the text to be mined

``` r
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower)) #making all text low caps
```

Using SnowballC package to remove some of the useless features of our reviews such as numbers, basic words like "the", "it", "a" etc. and also eliminating past tenses and punctuations. to keep only the roots of words. This is particularly useful when we match words to see if they mean the same. The last line reomves any extra blank spaces that will be created due to the elimination of useless features

``` r
library(SnowballC)
corpus = tm_map(corpus, removeNumbers) #to remove all numbers in reviews
corpus = tm_map(corpus, removePunctuation) #remove all punctuations
corpus = tm_map(corpus, removeWords, stopwords()) #remove useless words. You can use built-in stock words list
corpus = tm_map(corpus, stemDocument) #to do steming, remove tenses and focus on roots only
corpus = tm_map(corpus, stripWhitespace)#remove extra spaces
```

Now let us implement a bag of words model to create a document term matrix which is basicaly a collection of columns with one column for each word and one row being a review. For every word that is present in the review, the column will have a value of 1.

``` r
dtm = DocumentTermMatrix(corpus)
dtm
```

    ## <<DocumentTermMatrix (documents: 1000, terms: 1577)>>
    ## Non-/sparse entries: 5435/1571565
    ## Sparsity           : 100%
    ## Maximal term length: 32
    ## Weighting          : term frequency (tf)

The dtm is created above. It might be noted that the dtm might have several words which occur just once or twice and can be termed as sparse keywords. These words are of very less importance to our analysis

``` r
dtm = removeSparseTerms(dtm, 0.999) #filter non frequent words. Keep 99.9% of the words
#these words are non frequent and hence not relevant
dtm
```

    ## <<DocumentTermMatrix (documents: 1000, terms: 691)>>
    ## Non-/sparse entries: 4549/686451
    ## Sparsity           : 99%
    ## Maximal term length: 12
    ## Weighting          : term frequency (tf)

Now we ca move o to build our classifier. But before that we need to covernt out dtm to a data frame from a matrix

``` r
dataset = as.data.frame(as.matrix(dtm))
head(dataset[2:5]) #limiting the columns shown for aesthetic reasons
```

    ##   acknowledg actual ago almost
    ## 1          0      0   0      0
    ## 2          0      0   0      0
    ## 3          0      0   0      0
    ## 4          0      0   0      0
    ## 5          0      0   0      0
    ## 6          0      0   0      0

The colum headers we see above are words that have bee extracted as sigificat from our analysis. Let's try to build a wordcloud from these

``` r
library(wordcloud)
col_words <- names(dataset)
#col_words
wordcloud(col_words, max.words = 20, scale=c(3,.2), random.order = FALSE, colors = "black")
```

![](resturant_reviews_analysis_files/figure-markdown_github/unnamed-chunk-7-1.png)

Wonderful!

Let's move on to build a classifier Before I move on, let me add the rating column to the newly build dtm, which will be our dependent variable to predict and convert it into a factor

``` r
dataset$Liked = dataset_original$Liked
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))
head(dataset[1:4], 2)
```

    ##   absolut acknowledg actual ago
    ## 1       0          0      0   0
    ## 2       0          0      0   0

``` r
ncol(dataset)
```

    ## [1] 692

Let's now split the dataset into trainig ad test set with a 8/10 ratio

``` r
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
nrow(training_set)
```

    ## [1] 800

``` r
nrow(test_set)
```

    ## [1] 200

Perfect. We have 800 entries in the training set and 200 in the test set

Now let's build a Random Forest classifier to model the data with 10 trees. be careful while choosing ntree since higher might be good but lead to overfitting. Overfitting will lead to some poor predictions

In the code below 692 is the column index of the dependent variable and that has been removed while building our model

``` r
library(randomForest)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)
```

All good so far. We have our classifier ready. Lets move on and predict the test set

``` r
text_pred = predict(classifier,newdata = test_set[-692])
```

Let's check the confusion matrix and accuracy to evaluate how well our predictions have been

``` r
cm = table(test_set[, 692], text_pred)
cm
```

    ##    text_pred
    ##      0  1
    ##   0 79 21
    ##   1 30 70

``` r
accuracy <- (cm[1,1] + cm[2,2]) / (cm[1,1] +cm[2,1]+cm[2,2]+cm[1,2])*100
cat("accuracy =", accuracy)
```

    ## accuracy = 74.5

Definitely our results are not upto the mark here. 74.5% is just decent and we will have to improve our algorithm - may be use a K-fold cross validation to identify the best model and use grid search to optimize hyperparameters. Leaving that for another day...
