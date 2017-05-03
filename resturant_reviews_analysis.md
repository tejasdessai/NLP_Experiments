I am creating a classification engine that would read a set of reviews left on the webpage of a resturant and predict whether the reviews are positive or negative. The dataset has been divided into a training and test set to build and test the model.

Loading the text mining package and loading data

``` r
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)
#install.packages('tm') #text mining package
library(tm)
```

    ## Loading required package: NLP

``` r
head(dataset_original)
```

    ##                                                                                    Review
    ## 1                                                                Wow... Loved this place.
    ## 2                                                                      Crust is not good.
    ## 3                                               Not tasty and the texture was just nasty.
    ## 4 Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.
    ## 5                             The selection on the menu was great and so were the prices.
    ## 6                                          Now I am getting angry and I want my damn pho.
    ##   Liked
    ## 1     1
    ## 2     0
    ## 3     0
    ## 4     1
    ## 5     1
    ## 6     0

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
head(dataset)
```

    ##   absolut acknowledg actual ago almost also although alway amaz ambianc
    ## 1       0          0      0   0      0    0        0     0    0       0
    ## 2       0          0      0   0      0    0        0     0    0       0
    ## 3       0          0      0   0      0    0        0     0    0       0
    ## 4       0          0      0   0      0    0        0     0    0       0
    ## 5       0          0      0   0      0    0        0     0    0       0
    ## 6       0          0      0   0      0    0        0     0    0       0
    ##   ambienc amount anoth anyon anyth anytim anyway apolog appet area arent
    ## 1       0      0     0     0     0      0      0      0     0    0     0
    ## 2       0      0     0     0     0      0      0      0     0    0     0
    ## 3       0      0     0     0     0      0      0      0     0    0     0
    ## 4       0      0     0     0     0      0      0      0     0    0     0
    ## 5       0      0     0     0     0      0      0      0     0    0     0
    ## 6       0      0     0     0     0      0      0      0     0    0     0
    ##   around arriv ask assur ate atmospher attack attent attitud authent
    ## 1      0     0   0     0   0         0      0      0       0       0
    ## 2      0     0   0     0   0         0      0      0       0       0
    ## 3      0     0   0     0   0         0      0      0       0       0
    ## 4      0     0   0     0   0         0      0      0       0       0
    ## 5      0     0   0     0   0         0      0      0       0       0
    ## 6      0     0   0     0   0         0      0      0       0       0
    ##   averag avoid away awesom awkward babi bachi back bacon bad bagel bakeri
    ## 1      0     0    0      0       0    0     0    0     0   0     0      0
    ## 2      0     0    0      0       0    0     0    0     0   0     0      0
    ## 3      0     0    0      0       0    0     0    0     0   0     0      0
    ## 4      0     0    0      0       0    0     0    0     0   0     0      0
    ## 5      0     0    0      0       0    0     0    0     0   0     0      0
    ## 6      0     0    0      0       0    0     0    0     0   0     0      0
    ##   bar bare bartend basic bathroom batter bay bean beat beauti becom beef
    ## 1   0    0       0     0        0      0   0    0    0      0     0    0
    ## 2   0    0       0     0        0      0   0    0    0      0     0    0
    ## 3   0    0       0     0        0      0   0    0    0      0     0    0
    ## 4   0    0       0     0        0      0   0    0    0      0     0    0
    ## 5   0    0       0     0        0      0   0    0    0      0     0    0
    ## 6   0    0       0     0        0      0   0    0    0      0     0    0
    ##   beer behind believ belli best better beyond big bill biscuit bisqu bit
    ## 1    0      0      0     0    0      0      0   0    0       0     0   0
    ## 2    0      0      0     0    0      0      0   0    0       0     0   0
    ## 3    0      0      0     0    0      0      0   0    0       0     0   0
    ## 4    0      0      0     0    0      0      0   0    0       0     0   0
    ## 5    0      0      0     0    0      0      0   0    0       0     0   0
    ## 6    0      0      0     0    0      0      0   0    0       0     0   0
    ##   bite black bland blow boba boot bother bowl box boy boyfriend bread
    ## 1    0     0     0    0    0    0      0    0   0   0         0     0
    ## 2    0     0     0    0    0    0      0    0   0   0         0     0
    ## 3    0     0     0    0    0    0      0    0   0   0         0     0
    ## 4    0     0     0    0    0    0      0    0   0   0         0     0
    ## 5    0     0     0    0    0    0      0    0   0   0         0     0
    ## 6    0     0     0    0    0    0      0    0   0   0         0     0
    ##   break breakfast brick bring brought brunch buck buffet build burger busi
    ## 1     0         0     0     0       0      0    0      0     0      0    0
    ## 2     0         0     0     0       0      0    0      0     0      0    0
    ## 3     0         0     0     0       0      0    0      0     0      0    0
    ## 4     0         0     0     0       0      0    0      0     0      0    0
    ## 5     0         0     0     0       0      0    0      0     0      0    0
    ## 6     0         0     0     0       0      0    0      0     0      0    0
    ##   butter cafe call came can cant car care cashier char charcoal charg
    ## 1      0    0    0    0   0    0   0    0       0    0        0     0
    ## 2      0    0    0    0   0    0   0    0       0    0        0     0
    ## 3      0    0    0    0   0    0   0    0       0    0        0     0
    ## 4      0    0    0    0   0    0   0    0       0    0        0     0
    ## 5      0    0    0    0   0    0   0    0       0    0        0     0
    ## 6      0    0    0    0   0    0   0    0       0    0        0     0
    ##   cheap check chees cheeseburg chef chewi chicken chines chip choos
    ## 1     0     0     0          0    0     0       0      0    0     0
    ## 2     0     0     0          0    0     0       0      0    0     0
    ## 3     0     0     0          0    0     0       0      0    0     0
    ## 4     0     0     0          0    0     0       0      0    0     0
    ## 5     0     0     0          0    0     0       0      0    0     0
    ## 6     0     0     0          0    0     0       0      0    0     0
    ##   classic clean close cocktail coffe cold color combin combo come comfort
    ## 1       0     0     0        0     0    0     0      0     0    0       0
    ## 2       0     0     0        0     0    0     0      0     0    0       0
    ## 3       0     0     0        0     0    0     0      0     0    0       0
    ## 4       0     0     0        0     0    0     0      0     0    0       0
    ## 5       0     0     0        0     0    0     0      0     0    0       0
    ## 6       0     0     0        0     0    0     0      0     0    0       0
    ##   compani complain complaint complet consid contain conveni cook cool
    ## 1       0        0         0       0      0       0       0    0    0
    ## 2       0        0         0       0      0       0       0    0    0
    ## 3       0        0         0       0      0       0       0    0    0
    ## 4       0        0         0       0      0       0       0    0    0
    ## 5       0        0         0       0      0       0       0    0    0
    ## 6       0        0         0       0      0       0       0    0    0
    ##   correct couldnt coupl cours cover cow crazi cream creami crowd crust
    ## 1       0       0     0     0     0   0     0     0      0     0     0
    ## 2       0       0     0     0     0   0     0     0      0     0     1
    ## 3       0       0     0     0     0   0     0     0      0     0     0
    ## 4       0       0     0     0     0   0     0     0      0     0     0
    ## 5       0       0     0     0     0   0     0     0      0     0     0
    ## 6       0       0     0     0     0   0     0     0      0     0     0
    ##   curri custom cut cute damn dark date day deal decent decid decor defin
    ## 1     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 2     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 3     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 4     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 5     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 6     0      0   0    0    1    0    0   0    0      0     0     0     0
    ##   definit delici delight delish deserv dessert didnt die differ dine
    ## 1       0      0       0      0      0       0     0   0      0    0
    ## 2       0      0       0      0      0       0     0   0      0    0
    ## 3       0      0       0      0      0       0     0   0      0    0
    ## 4       0      0       0      0      0       0     0   0      0    0
    ## 5       0      0       0      0      0       0     0   0      0    0
    ## 6       0      0       0      0      0       0     0   0      0    0
    ##   dinner dirt dirti disappoint disgrac disgust dish disrespect dog done
    ## 1      0    0     0          0       0       0    0          0   0    0
    ## 2      0    0     0          0       0       0    0          0   0    0
    ## 3      0    0     0          0       0       0    0          0   0    0
    ## 4      0    0     0          0       0       0    0          0   0    0
    ## 5      0    0     0          0       0       0    0          0   0    0
    ## 6      0    0     0          0       0       0    0          0   0    0
    ##   dont door doubl doubt downtown dress dri driest drink drive duck eat
    ## 1    0    0     0     0        0     0   0      0     0     0    0   0
    ## 2    0    0     0     0        0     0   0      0     0     0    0   0
    ## 3    0    0     0     0        0     0   0      0     0     0    0   0
    ## 4    0    0     0     0        0     0   0      0     0     0    0   0
    ## 5    0    0     0     0        0     0   0      0     0     0    0   0
    ## 6    0    0     0     0        0     0   0      0     0     0    0   0
    ##   eaten edibl egg eggplant either els elsewher employe empti end enjoy
    ## 1     0     0   0        0      0   0        0       0     0   0     0
    ## 2     0     0   0        0      0   0        0       0     0   0     0
    ## 3     0     0   0        0      0   0        0       0     0   0     0
    ## 4     0     0   0        0      0   0        0       0     0   0     0
    ## 5     0     0   0        0      0   0        0       0     0   0     0
    ## 6     0     0   0        0      0   0        0       0     0   0     0
    ##   enough entre equal especi establish even event ever everi everyon
    ## 1      0     0     0      0         0    0     0    0     0       0
    ## 2      0     0     0      0         0    0     0    0     0       0
    ## 3      0     0     0      0         0    0     0    0     0       0
    ## 4      0     0     0      0         0    0     0    0     0       0
    ## 5      0     0     0      0         0    0     0    0     0       0
    ## 6      0     0     0      0         0    0     0    0     0       0
    ##   everyth excel excus expect experi experienc extra extrem eye fact fail
    ## 1       0     0     0      0      0         0     0      0   0    0    0
    ## 2       0     0     0      0      0         0     0      0   0    0    0
    ## 3       0     0     0      0      0         0     0      0   0    0    0
    ## 4       0     0     0      0      0         0     0      0   0    0    0
    ## 5       0     0     0      0      0         0     0      0   0    0    0
    ## 6       0     0     0      0      0         0     0      0   0    0    0
    ##   fair famili familiar fan fantast far fare fast favor favorit feel fell
    ## 1    0      0        0   0       0   0    0    0     0       0    0    0
    ## 2    0      0        0   0       0   0    0    0     0       0    0    0
    ## 3    0      0        0   0       0   0    0    0     0       0    0    0
    ## 4    0      0        0   0       0   0    0    0     0       0    0    0
    ## 5    0      0        0   0       0   0    0    0     0       0    0    0
    ## 6    0      0        0   0       0   0    0    0     0       0    0    0
    ##   felt filet fill final find fine finish first fish flavor flavorless
    ## 1    0     0    0     0    0    0      0     0    0      0          0
    ## 2    0     0    0     0    0    0      0     0    0      0          0
    ## 3    0     0    0     0    0    0      0     0    0      0          0
    ## 4    0     0    0     0    0    0      0     0    0      0          0
    ## 5    0     0    0     0    0    0      0     0    0      0          0
    ## 6    0     0    0     0    0    0      0     0    0      0          0
    ##   flower focus folk food found fresh fri friend front frozen full fun
    ## 1      0     0    0    0     0     0   0      0     0      0    0   0
    ## 2      0     0    0    0     0     0   0      0     0      0    0   0
    ## 3      0     0    0    0     0     0   0      0     0      0    0   0
    ## 4      0     0    0    0     0     0   0      0     0      0    0   0
    ## 5      0     0    0    0     0     0   0      0     0      0    0   0
    ## 6      0     0    0    0     0     0   0      0     0      0    0   0
    ##   garlic gave generous get give given glad gold gone good got greas great
    ## 1      0    0        0   0    0     0    0    0    0    0   0     0     0
    ## 2      0    0        0   0    0     0    0    0    0    1   0     0     0
    ## 3      0    0        0   0    0     0    0    0    0    0   0     0     0
    ## 4      0    0        0   0    0     0    0    0    0    0   0     0     0
    ## 5      0    0        0   0    0     0    0    0    0    0   0     0     1
    ## 6      0    0        0   1    0     0    0    0    0    0   0     0     0
    ##   greek green greet grill gross group guess guest guy gyro hair half hand
    ## 1     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 2     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 3     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 4     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 5     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 6     0     0     0     0     0     0     0     0   0    0    0    0    0
    ##   handl happen happi hard hate head healthi heard heart heat help high
    ## 1     0      0     0    0    0    0       0     0     0    0    0    0
    ## 2     0      0     0    0    0    0       0     0     0    0    0    0
    ## 3     0      0     0    0    0    0       0     0     0    0    0    0
    ## 4     0      0     0    0    0    0       0     0     0    0    0    0
    ## 5     0      0     0    0    0    0       0     0     0    0    0    0
    ## 6     0      0     0    0    0    0       0     0     0    0    0    0
    ##   highlight hit home homemad honest hope horribl hot hour hous howev huge
    ## 1         0   0    0       0      0    0       0   0    0    0     0    0
    ## 2         0   0    0       0      0    0       0   0    0    0     0    0
    ## 3         0   0    0       0      0    0       0   0    0    0     0    0
    ## 4         0   0    0       0      0    0       0   0    0    0     0    0
    ## 5         0   0    0       0      0    0       0   0    0    0     0    0
    ## 6         0   0    0       0      0    0       0   0    0    0     0    0
    ##   human hummus husband ice ignor ill imagin immedi impecc impress includ
    ## 1     0      0       0   0     0   0      0      0      0       0      0
    ## 2     0      0       0   0     0   0      0      0      0       0      0
    ## 3     0      0       0   0     0   0      0      0      0       0      0
    ## 4     0      0       0   0     0   0      0      0      0       0      0
    ## 5     0      0       0   0     0   0      0      0      0       0      0
    ## 6     0      0       0   0     0   0      0      0      0       0      0
    ##   incred indian inexpens insid insult interest isnt italian ive job joint
    ## 1      0      0        0     0      0        0    0       0   0   0     0
    ## 2      0      0        0     0      0        0    0       0   0   0     0
    ## 3      0      0        0     0      0        0    0       0   0   0     0
    ## 4      0      0        0     0      0        0    0       0   0   0     0
    ## 5      0      0        0     0      0        0    0       0   0   0     0
    ## 6      0      0        0     0      0        0    0       0   0   0     0
    ##   joke judg just kept kid kind know known lack ladi larg last late later
    ## 1    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ## 2    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ## 3    0    0    1    0   0    0    0     0    0    0    0    0    0     0
    ## 4    0    0    0    0   0    0    0     0    0    0    0    0    1     0
    ## 5    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ## 6    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ##   least leav left legit let life light like list liter littl live lobster
    ## 1     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 2     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 3     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 4     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 5     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 6     0    0    0     0   0    0     0    0    0     0     0    0       0
    ##   locat long longer look lost lot love lover lukewarm lunch made main make
    ## 1     0    0      0    0    0   0    1     0        0     0    0    0    0
    ## 2     0    0      0    0    0   0    0     0        0     0    0    0    0
    ## 3     0    0      0    0    0   0    0     0        0     0    0    0    0
    ## 4     0    0      0    0    0   0    1     0        0     0    0    0    0
    ## 5     0    0      0    0    0   0    0     0        0     0    0    0    0
    ## 6     0    0      0    0    0   0    0     0        0     0    0    0    0
    ##   mall manag mani margarita mari may mayb meal mean meat mediocr meh melt
    ## 1    0     0    0         0    0   0    0    0    0    0       0   0    0
    ## 2    0     0    0         0    0   0    0    0    0    0       0   0    0
    ## 3    0     0    0         0    0   0    0    0    0    0       0   0    0
    ## 4    0     0    0         0    0   1    0    0    0    0       0   0    0
    ## 5    0     0    0         0    0   0    0    0    0    0       0   0    0
    ## 6    0     0    0         0    0   0    0    0    0    0       0   0    0
    ##   menu mexican mid min mind minut miss mistak moist mom money mood mouth
    ## 1    0       0   0   0    0     0    0      0     0   0     0    0     0
    ## 2    0       0   0   0    0     0    0      0     0   0     0    0     0
    ## 3    0       0   0   0    0     0    0      0     0   0     0    0     0
    ## 4    0       0   0   0    0     0    0      0     0   0     0    0     0
    ## 5    1       0   0   0    0     0    0      0     0   0     0    0     0
    ## 6    0       0   0   0    0     0    0      0     0   0     0    0     0
    ##   much multipl mushroom music must nacho nasti need needless neighborhood
    ## 1    0       0        0     0    0     0     0    0        0            0
    ## 2    0       0        0     0    0     0     0    0        0            0
    ## 3    0       0        0     0    0     0     1    0        0            0
    ## 4    0       0        0     0    0     0     0    0        0            0
    ## 5    0       0        0     0    0     0     0    0        0            0
    ## 6    0       0        0     0    0     0     0    0        0            0
    ##   never new next nice nicest night none note noth now offer old omg one
    ## 1     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 2     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 3     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 4     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 5     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 6     0   0    0    0      0     0    0    0    0   1     0   0   0   0
    ##   opportun option order other outsid outstand oven overal overcook overpr
    ## 1        0      0     0     0      0        0    0      0        0      0
    ## 2        0      0     0     0      0        0    0      0        0      0
    ## 3        0      0     0     0      0        0    0      0        0      0
    ## 4        0      0     0     0      0        0    0      0        0      0
    ## 5        0      0     0     0      0        0    0      0        0      0
    ## 6        0      0     0     0      0        0    0      0        0      0
    ##   overwhelm owner pace pack paid pancak paper par part parti pass pasta
    ## 1         0     0    0    0    0      0     0   0    0     0    0     0
    ## 2         0     0    0    0    0      0     0   0    0     0    0     0
    ## 3         0     0    0    0    0      0     0   0    0     0    0     0
    ## 4         0     0    0    0    0      0     0   0    0     0    0     0
    ## 5         0     0    0    0    0      0     0   0    0     0    0     0
    ## 6         0     0    0    0    0      0     0   0    0     0    0     0
    ##   patio pay peanut peopl perfect person pho phoenix pictur piec pita pizza
    ## 1     0   0      0     0       0      0   0       0      0    0    0     0
    ## 2     0   0      0     0       0      0   0       0      0    0    0     0
    ## 3     0   0      0     0       0      0   0       0      0    0    0     0
    ## 4     0   0      0     0       0      0   0       0      0    0    0     0
    ## 5     0   0      0     0       0      0   0       0      0    0    0     0
    ## 6     0   0      0     0       0      0   1       0      0    0    0     0
    ##   place plate play pleas pleasant plus point poor pop pork portion possibl
    ## 1     1     0    0     0        0    0     0    0   0    0       0       0
    ## 2     0     0    0     0        0    0     0    0   0    0       0       0
    ## 3     0     0    0     0        0    0     0    0   0    0       0       0
    ## 4     0     0    0     0        0    0     0    0   0    0       0       0
    ## 5     0     0    0     0        0    0     0    0   0    0       0       0
    ## 6     0     0    0     0        0    0     0    0   0    0       0       0
    ##   potato prepar present pretti price probabl profession promis prompt
    ## 1      0      0       0      0     0       0          0      0      0
    ## 2      0      0       0      0     0       0          0      0      0
    ## 3      0      0       0      0     0       0          0      0      0
    ## 4      0      0       0      0     0       0          0      0      0
    ## 5      0      0       0      0     1       0          0      0      0
    ## 6      0      0       0      0     0       0          0      0      0
    ##   provid public pull pure put qualiti quick quit rare rate rather rave
    ## 1      0      0    0    0   0       0     0    0    0    0      0    0
    ## 2      0      0    0    0   0       0     0    0    0    0      0    0
    ## 3      0      0    0    0   0       0     0    0    0    0      0    0
    ## 4      0      0    0    0   0       0     0    0    0    0      0    0
    ## 5      0      0    0    0   0       0     0    0    0    0      0    0
    ## 6      0      0    0    0   0       0     0    0    0    0      0    0
    ##   read real realiz realli reason receiv recent recommend red refil regular
    ## 1    0    0      0      0      0      0      0         0   0     0       0
    ## 2    0    0      0      0      0      0      0         0   0     0       0
    ## 3    0    0      0      0      0      0      0         0   0     0       0
    ## 4    0    0      0      0      0      0      0         1   0     0       0
    ## 5    0    0      0      0      0      0      0         0   0     0       0
    ## 6    0    0      0      0      0      0      0         0   0     0       0
    ##   relax remind restaur return review rice right roast roll room rude run
    ## 1     0      0       0      0      0    0     0     0    0    0    0   0
    ## 2     0      0       0      0      0    0     0     0    0    0    0   0
    ## 3     0      0       0      0      0    0     0     0    0    0    0   0
    ## 4     0      0       0      0      0    0     0     0    0    0    0   0
    ## 5     0      0       0      0      0    0     0     0    0    0    0   0
    ## 6     0      0       0      0      0    0     0     0    0    0    0   0
    ##   sad said salad salmon salsa salt sandwich sashimi sat satisfi sauc say
    ## 1   0    0     0      0     0    0        0       0   0       0    0   0
    ## 2   0    0     0      0     0    0        0       0   0       0    0   0
    ## 3   0    0     0      0     0    0        0       0   0       0    0   0
    ## 4   0    0     0      0     0    0        0       0   0       0    0   0
    ## 5   0    0     0      0     0    0        0       0   0       0    0   0
    ## 6   0    0     0      0     0    0        0       0   0       0    0   0
    ##   scallop seafood season seat second see seem seen select serious serv
    ## 1       0       0      0    0      0   0    0    0      0       0    0
    ## 2       0       0      0    0      0   0    0    0      0       0    0
    ## 3       0       0      0    0      0   0    0    0      0       0    0
    ## 4       0       0      0    0      0   0    0    0      0       0    0
    ## 5       0       0      0    0      0   0    0    0      1       0    0
    ## 6       0       0      0    0      0   0    0    0      0       0    0
    ##   server servic set sever shop show shrimp sick side sign similar simpl
    ## 1      0      0   0     0    0    0      0    0    0    0       0     0
    ## 2      0      0   0     0    0    0      0    0    0    0       0     0
    ## 3      0      0   0     0    0    0      0    0    0    0       0     0
    ## 4      0      0   0     0    0    0      0    0    0    0       0     0
    ## 5      0      0   0     0    0    0      0    0    0    0       0     0
    ## 6      0      0   0     0    0    0      0    0    0    0       0     0
    ##   simpli sinc singl sit six slice slow small smell soggi someon someth
    ## 1      0    0     0   0   0     0    0     0     0     0      0      0
    ## 2      0    0     0   0   0     0    0     0     0     0      0      0
    ## 3      0    0     0   0   0     0    0     0     0     0      0      0
    ## 4      0    0     0   0   0     0    0     0     0     0      0      0
    ## 5      0    0     0   0   0     0    0     0     0     0      0      0
    ## 6      0    0     0   0   0     0    0     0     0     0      0      0
    ##   soon soooo sore soup special spend spice spici spot staff stale star
    ## 1    0     0    0    0       0     0     0     0    0     0     0    0
    ## 2    0     0    0    0       0     0     0     0    0     0     0    0
    ## 3    0     0    0    0       0     0     0     0    0     0     0    0
    ## 4    0     0    0    0       0     0     0     0    0     0     0    0
    ## 5    0     0    0    0       0     0     0     0    0     0     0    0
    ## 6    0     0    0    0       0     0     0     0    0     0     0    0
    ##   start station stay steak step stick still stir stomach stop strip stuf
    ## 1     0       0    0     0    0     0     0    0       0    0     0    0
    ## 2     0       0    0     0    0     0     0    0       0    0     0    0
    ## 3     0       0    0     0    0     0     0    0       0    0     0    0
    ## 4     0       0    0     0    0     0     0    0       0    1     0    0
    ## 5     0       0    0     0    0     0     0    0       0    0     0    0
    ## 6     0       0    0     0    0     0     0    0       0    0     0    0
    ##   stuff style subpar subway suck sugari suggest summer super sure surpris
    ## 1     0     0      0      0    0      0       0      0     0    0       0
    ## 2     0     0      0      0    0      0       0      0     0    0       0
    ## 3     0     0      0      0    0      0       0      0     0    0       0
    ## 4     0     0      0      0    0      0       0      0     0    0       0
    ## 5     0     0      0      0    0      0       0      0     0    0       0
    ## 6     0     0      0      0    0      0       0      0     0    0       0
    ##   sushi sweet tabl taco take talk tap tapa tartar tast tasteless tasti tea
    ## 1     0     0    0    0    0    0   0    0      0    0         0     0   0
    ## 2     0     0    0    0    0    0   0    0      0    0         0     0   0
    ## 3     0     0    0    0    0    0   0    0      0    0         0     1   0
    ## 4     0     0    0    0    0    0   0    0      0    0         0     0   0
    ## 5     0     0    0    0    0    0   0    0      0    0         0     0   0
    ## 6     0     0    0    0    0    0   0    0      0    0         0     0   0
    ##   tell ten tender terribl textur thai that thin thing think third though
    ## 1    0   0      0       0      0    0    0    0     0     0     0      0
    ## 2    0   0      0       0      0    0    0    0     0     0     0      0
    ## 3    0   0      0       0      1    0    0    0     0     0     0      0
    ## 4    0   0      0       0      0    0    0    0     0     0     0      0
    ## 5    0   0      0       0      0    0    0    0     0     0     0      0
    ## 6    0   0      0       0      0    0    0    0     0     0     0      0
    ##   thought thumb time tip toast today told took top tot total touch toward
    ## 1       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 2       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 3       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 4       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 5       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 6       0     0    0   0     0     0    0    0   0   0     0     0      0
    ##   town treat tri trip tuna twice two unbeliev undercook underwhelm
    ## 1    0     0   0    0    0     0   0        0         0          0
    ## 2    0     0   0    0    0     0   0        0         0          0
    ## 3    0     0   0    0    0     0   0        0         0          0
    ## 4    0     0   0    0    0     0   0        0         0          0
    ## 5    0     0   0    0    0     0   0        0         0          0
    ## 6    0     0   0    0    0     0   0        0         0          0
    ##   unfortun unless use valley valu vega veget vegetarian ventur vibe
    ## 1        0      0   0      0    0    0     0          0      0    0
    ## 2        0      0   0      0    0    0     0          0      0    0
    ## 3        0      0   0      0    0    0     0          0      0    0
    ## 4        0      0   0      0    0    0     0          0      0    0
    ## 5        0      0   0      0    0    0     0          0      0    0
    ## 6        0      0   0      0    0    0     0          0      0    0
    ##   vinegrett visit wait waiter waitress walk wall want warm wasnt wast
    ## 1         0     0    0      0        0    0    0    0    0     0    0
    ## 2         0     0    0      0        0    0    0    0    0     0    0
    ## 3         0     0    0      0        0    0    0    0    0     0    0
    ## 4         0     0    0      0        0    0    0    0    0     0    0
    ## 5         0     0    0      0        0    0    0    0    0     0    0
    ## 6         0     0    0      0        0    0    0    1    0     0    0
    ##   watch water way week well went weve white whole wife will wine wing
    ## 1     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 2     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 3     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 4     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 5     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 6     0     0   0    0    0    0    0     0     0    0    0    0    0
    ##   without wonder wont word work worker world wors worst worth wouldnt wow
    ## 1       0      0    0    0    0      0     0    0     0     0       0   1
    ## 2       0      0    0    0    0      0     0    0     0     0       0   0
    ## 3       0      0    0    0    0      0     0    0     0     0       0   0
    ## 4       0      0    0    0    0      0     0    0     0     0       0   0
    ## 5       0      0    0    0    0      0     0    0     0     0       0   0
    ## 6       0      0    0    0    0      0     0    0     0     0       0   0
    ##   wrap wrong year yet youd your yummi zero
    ## 1    0     0    0   0    0    0     0    0
    ## 2    0     0    0   0    0    0     0    0
    ## 3    0     0    0   0    0    0     0    0
    ## 4    0     0    0   0    0    0     0    0
    ## 5    0     0    0   0    0    0     0    0
    ## 6    0     0    0   0    0    0     0    0

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
head(dataset, 2)
```

    ##   absolut acknowledg actual ago almost also although alway amaz ambianc
    ## 1       0          0      0   0      0    0        0     0    0       0
    ## 2       0          0      0   0      0    0        0     0    0       0
    ##   ambienc amount anoth anyon anyth anytim anyway apolog appet area arent
    ## 1       0      0     0     0     0      0      0      0     0    0     0
    ## 2       0      0     0     0     0      0      0      0     0    0     0
    ##   around arriv ask assur ate atmospher attack attent attitud authent
    ## 1      0     0   0     0   0         0      0      0       0       0
    ## 2      0     0   0     0   0         0      0      0       0       0
    ##   averag avoid away awesom awkward babi bachi back bacon bad bagel bakeri
    ## 1      0     0    0      0       0    0     0    0     0   0     0      0
    ## 2      0     0    0      0       0    0     0    0     0   0     0      0
    ##   bar bare bartend basic bathroom batter bay bean beat beauti becom beef
    ## 1   0    0       0     0        0      0   0    0    0      0     0    0
    ## 2   0    0       0     0        0      0   0    0    0      0     0    0
    ##   beer behind believ belli best better beyond big bill biscuit bisqu bit
    ## 1    0      0      0     0    0      0      0   0    0       0     0   0
    ## 2    0      0      0     0    0      0      0   0    0       0     0   0
    ##   bite black bland blow boba boot bother bowl box boy boyfriend bread
    ## 1    0     0     0    0    0    0      0    0   0   0         0     0
    ## 2    0     0     0    0    0    0      0    0   0   0         0     0
    ##   break breakfast brick bring brought brunch buck buffet build burger busi
    ## 1     0         0     0     0       0      0    0      0     0      0    0
    ## 2     0         0     0     0       0      0    0      0     0      0    0
    ##   butter cafe call came can cant car care cashier char charcoal charg
    ## 1      0    0    0    0   0    0   0    0       0    0        0     0
    ## 2      0    0    0    0   0    0   0    0       0    0        0     0
    ##   cheap check chees cheeseburg chef chewi chicken chines chip choos
    ## 1     0     0     0          0    0     0       0      0    0     0
    ## 2     0     0     0          0    0     0       0      0    0     0
    ##   classic clean close cocktail coffe cold color combin combo come comfort
    ## 1       0     0     0        0     0    0     0      0     0    0       0
    ## 2       0     0     0        0     0    0     0      0     0    0       0
    ##   compani complain complaint complet consid contain conveni cook cool
    ## 1       0        0         0       0      0       0       0    0    0
    ## 2       0        0         0       0      0       0       0    0    0
    ##   correct couldnt coupl cours cover cow crazi cream creami crowd crust
    ## 1       0       0     0     0     0   0     0     0      0     0     0
    ## 2       0       0     0     0     0   0     0     0      0     0     1
    ##   curri custom cut cute damn dark date day deal decent decid decor defin
    ## 1     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 2     0      0   0    0    0    0    0   0    0      0     0     0     0
    ##   definit delici delight delish deserv dessert didnt die differ dine
    ## 1       0      0       0      0      0       0     0   0      0    0
    ## 2       0      0       0      0      0       0     0   0      0    0
    ##   dinner dirt dirti disappoint disgrac disgust dish disrespect dog done
    ## 1      0    0     0          0       0       0    0          0   0    0
    ## 2      0    0     0          0       0       0    0          0   0    0
    ##   dont door doubl doubt downtown dress dri driest drink drive duck eat
    ## 1    0    0     0     0        0     0   0      0     0     0    0   0
    ## 2    0    0     0     0        0     0   0      0     0     0    0   0
    ##   eaten edibl egg eggplant either els elsewher employe empti end enjoy
    ## 1     0     0   0        0      0   0        0       0     0   0     0
    ## 2     0     0   0        0      0   0        0       0     0   0     0
    ##   enough entre equal especi establish even event ever everi everyon
    ## 1      0     0     0      0         0    0     0    0     0       0
    ## 2      0     0     0      0         0    0     0    0     0       0
    ##   everyth excel excus expect experi experienc extra extrem eye fact fail
    ## 1       0     0     0      0      0         0     0      0   0    0    0
    ## 2       0     0     0      0      0         0     0      0   0    0    0
    ##   fair famili familiar fan fantast far fare fast favor favorit feel fell
    ## 1    0      0        0   0       0   0    0    0     0       0    0    0
    ## 2    0      0        0   0       0   0    0    0     0       0    0    0
    ##   felt filet fill final find fine finish first fish flavor flavorless
    ## 1    0     0    0     0    0    0      0     0    0      0          0
    ## 2    0     0    0     0    0    0      0     0    0      0          0
    ##   flower focus folk food found fresh fri friend front frozen full fun
    ## 1      0     0    0    0     0     0   0      0     0      0    0   0
    ## 2      0     0    0    0     0     0   0      0     0      0    0   0
    ##   garlic gave generous get give given glad gold gone good got greas great
    ## 1      0    0        0   0    0     0    0    0    0    0   0     0     0
    ## 2      0    0        0   0    0     0    0    0    0    1   0     0     0
    ##   greek green greet grill gross group guess guest guy gyro hair half hand
    ## 1     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 2     0     0     0     0     0     0     0     0   0    0    0    0    0
    ##   handl happen happi hard hate head healthi heard heart heat help high
    ## 1     0      0     0    0    0    0       0     0     0    0    0    0
    ## 2     0      0     0    0    0    0       0     0     0    0    0    0
    ##   highlight hit home homemad honest hope horribl hot hour hous howev huge
    ## 1         0   0    0       0      0    0       0   0    0    0     0    0
    ## 2         0   0    0       0      0    0       0   0    0    0     0    0
    ##   human hummus husband ice ignor ill imagin immedi impecc impress includ
    ## 1     0      0       0   0     0   0      0      0      0       0      0
    ## 2     0      0       0   0     0   0      0      0      0       0      0
    ##   incred indian inexpens insid insult interest isnt italian ive job joint
    ## 1      0      0        0     0      0        0    0       0   0   0     0
    ## 2      0      0        0     0      0        0    0       0   0   0     0
    ##   joke judg just kept kid kind know known lack ladi larg last late later
    ## 1    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ## 2    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ##   least leav left legit let life light like list liter littl live lobster
    ## 1     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 2     0    0    0     0   0    0     0    0    0     0     0    0       0
    ##   locat long longer look lost lot love lover lukewarm lunch made main make
    ## 1     0    0      0    0    0   0    1     0        0     0    0    0    0
    ## 2     0    0      0    0    0   0    0     0        0     0    0    0    0
    ##   mall manag mani margarita mari may mayb meal mean meat mediocr meh melt
    ## 1    0     0    0         0    0   0    0    0    0    0       0   0    0
    ## 2    0     0    0         0    0   0    0    0    0    0       0   0    0
    ##   menu mexican mid min mind minut miss mistak moist mom money mood mouth
    ## 1    0       0   0   0    0     0    0      0     0   0     0    0     0
    ## 2    0       0   0   0    0     0    0      0     0   0     0    0     0
    ##   much multipl mushroom music must nacho nasti need needless neighborhood
    ## 1    0       0        0     0    0     0     0    0        0            0
    ## 2    0       0        0     0    0     0     0    0        0            0
    ##   never new next nice nicest night none note noth now offer old omg one
    ## 1     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 2     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ##   opportun option order other outsid outstand oven overal overcook overpr
    ## 1        0      0     0     0      0        0    0      0        0      0
    ## 2        0      0     0     0      0        0    0      0        0      0
    ##   overwhelm owner pace pack paid pancak paper par part parti pass pasta
    ## 1         0     0    0    0    0      0     0   0    0     0    0     0
    ## 2         0     0    0    0    0      0     0   0    0     0    0     0
    ##   patio pay peanut peopl perfect person pho phoenix pictur piec pita pizza
    ## 1     0   0      0     0       0      0   0       0      0    0    0     0
    ## 2     0   0      0     0       0      0   0       0      0    0    0     0
    ##   place plate play pleas pleasant plus point poor pop pork portion possibl
    ## 1     1     0    0     0        0    0     0    0   0    0       0       0
    ## 2     0     0    0     0        0    0     0    0   0    0       0       0
    ##   potato prepar present pretti price probabl profession promis prompt
    ## 1      0      0       0      0     0       0          0      0      0
    ## 2      0      0       0      0     0       0          0      0      0
    ##   provid public pull pure put qualiti quick quit rare rate rather rave
    ## 1      0      0    0    0   0       0     0    0    0    0      0    0
    ## 2      0      0    0    0   0       0     0    0    0    0      0    0
    ##   read real realiz realli reason receiv recent recommend red refil regular
    ## 1    0    0      0      0      0      0      0         0   0     0       0
    ## 2    0    0      0      0      0      0      0         0   0     0       0
    ##   relax remind restaur return review rice right roast roll room rude run
    ## 1     0      0       0      0      0    0     0     0    0    0    0   0
    ## 2     0      0       0      0      0    0     0     0    0    0    0   0
    ##   sad said salad salmon salsa salt sandwich sashimi sat satisfi sauc say
    ## 1   0    0     0      0     0    0        0       0   0       0    0   0
    ## 2   0    0     0      0     0    0        0       0   0       0    0   0
    ##   scallop seafood season seat second see seem seen select serious serv
    ## 1       0       0      0    0      0   0    0    0      0       0    0
    ## 2       0       0      0    0      0   0    0    0      0       0    0
    ##   server servic set sever shop show shrimp sick side sign similar simpl
    ## 1      0      0   0     0    0    0      0    0    0    0       0     0
    ## 2      0      0   0     0    0    0      0    0    0    0       0     0
    ##   simpli sinc singl sit six slice slow small smell soggi someon someth
    ## 1      0    0     0   0   0     0    0     0     0     0      0      0
    ## 2      0    0     0   0   0     0    0     0     0     0      0      0
    ##   soon soooo sore soup special spend spice spici spot staff stale star
    ## 1    0     0    0    0       0     0     0     0    0     0     0    0
    ## 2    0     0    0    0       0     0     0     0    0     0     0    0
    ##   start station stay steak step stick still stir stomach stop strip stuf
    ## 1     0       0    0     0    0     0     0    0       0    0     0    0
    ## 2     0       0    0     0    0     0     0    0       0    0     0    0
    ##   stuff style subpar subway suck sugari suggest summer super sure surpris
    ## 1     0     0      0      0    0      0       0      0     0    0       0
    ## 2     0     0      0      0    0      0       0      0     0    0       0
    ##   sushi sweet tabl taco take talk tap tapa tartar tast tasteless tasti tea
    ## 1     0     0    0    0    0    0   0    0      0    0         0     0   0
    ## 2     0     0    0    0    0    0   0    0      0    0         0     0   0
    ##   tell ten tender terribl textur thai that thin thing think third though
    ## 1    0   0      0       0      0    0    0    0     0     0     0      0
    ## 2    0   0      0       0      0    0    0    0     0     0     0      0
    ##   thought thumb time tip toast today told took top tot total touch toward
    ## 1       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 2       0     0    0   0     0     0    0    0   0   0     0     0      0
    ##   town treat tri trip tuna twice two unbeliev undercook underwhelm
    ## 1    0     0   0    0    0     0   0        0         0          0
    ## 2    0     0   0    0    0     0   0        0         0          0
    ##   unfortun unless use valley valu vega veget vegetarian ventur vibe
    ## 1        0      0   0      0    0    0     0          0      0    0
    ## 2        0      0   0      0    0    0     0          0      0    0
    ##   vinegrett visit wait waiter waitress walk wall want warm wasnt wast
    ## 1         0     0    0      0        0    0    0    0    0     0    0
    ## 2         0     0    0      0        0    0    0    0    0     0    0
    ##   watch water way week well went weve white whole wife will wine wing
    ## 1     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 2     0     0   0    0    0    0    0     0     0    0    0    0    0
    ##   without wonder wont word work worker world wors worst worth wouldnt wow
    ## 1       0      0    0    0    0      0     0    0     0     0       0   1
    ## 2       0      0    0    0    0      0     0    0     0     0       0   0
    ##   wrap wrong year yet youd your yummi zero Liked
    ## 1    0     0    0   0    0    0     0    0     1
    ## 2    0     0    0   0    0    0     0    0     0

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

Definitely our results are not upto the mark here. 76.5% is just decent and we will have to improve our algorithm - may be use a K-fold cross validation to identify the best model and use grid search to optimize hyperparameters. Leaving that for another day...
