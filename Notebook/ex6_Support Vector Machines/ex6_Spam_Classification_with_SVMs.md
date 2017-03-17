Spam Classification with SVMs
================

Initialization
--------------

``` r
rm(list=ls())
sources <- c("emailFeatures.R","getVocabList.R","linearKernel.R","bsxfun.R",
             "processEmail.R","svmPredict.R","svmTrain.R","meshgrid.R")

for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}
```

    ## Loading  emailFeatures.R 
    ## Loading  getVocabList.R 
    ## Loading  linearKernel.R 
    ## Loading  bsxfun.R 
    ## Loading  processEmail.R 
    ## Loading  svmPredict.R 
    ## Loading  svmTrain.R 
    ## Loading  meshgrid.R

Part 1: Email Preprocessing
---------------------------

``` r
cat(sprintf('\nPreprocessing sample email (emailSample1.txt)\n'))
```

    ## 
    ## Preprocessing sample email (emailSample1.txt)

``` r
# Extract Features
fName <- 'emailSample1.txt'
file_contents <- readChar(fName,file.info(fName)$size)
```

    ## Warning in readChar(fName, file.info(fName)$size): 在non-UTF-8 MBCS语言环境
    ## 里只能读取字节

``` r
word_indices  <- processEmail(file_contents)
```

    ## 
    ## ----- Processed Email -----

    ## Loading required package: SnowballC

    ## anyon know how much it cost to host a web portal well it depend on how mani visitor you re expect thi can be anywher from less than number buck a month to a coupl of dollarnumb you should checkout httpaddr or perhap amazon ecnumb if your run someth big to unsubscrib yourself from thi mail list send an email to emailaddr
    ## 
    ## --------------------------

``` r
# Print Stats
cat(sprintf('Word Indices: \n'))
```

    ## Word Indices:

``` r
cat(sprintf(' %d', word_indices))
```

    ##  86  916  794  1077  883  370  1699  790  1822  1831  883  431  1171  794  1002  1893  1364  592  1676  238  162  89  688  945  1663  1120  1062  1699  375  1162  479  1893  1510  799  1182  1237  810  1895  1440  1547  181  1699  1758  1896  688  1676  992  961  1477  71  530  1699  531

``` r
cat(sprintf('\n\n'))
```

Part 2: Feature Extraction
--------------------------

``` r
cat(sprintf('\nExtracting features from sample email (emailSample1.txt)\n'))
```

    ## 
    ## Extracting features from sample email (emailSample1.txt)

``` r
# Extract Features
fName <- 'emailSample1.txt'
file_contents <- readChar(fName,file.info(fName)$size)
```

    ## Warning in readChar(fName, file.info(fName)$size): 在non-UTF-8 MBCS语言环境
    ## 里只能读取字节

``` r
word_indices  <- processEmail(file_contents)
```

    ## 
    ## ----- Processed Email -----
    ## 
    ## anyon know how much it cost to host a web portal well it depend on how mani visitor you re expect thi can be anywher from less than number buck a month to a coupl of dollarnumb you should checkout httpaddr or perhap amazon ecnumb if your run someth big to unsubscrib yourself from thi mail list send an email to emailaddr
    ## 
    ## --------------------------

``` r
features      <- emailFeatures(word_indices)

# Print Stats
cat(sprintf('Length of feature vector: %d\n', length(features)))
```

    ## Length of feature vector: 1899

``` r
cat(sprintf('Number of non-zero entries: %d\n', sum(features > 0)))
```

    ## Number of non-zero entries: 45

Part 3: Train Linear SVM for Spam Classification
------------------------------------------------

``` r
# Load the Spam Email dataset
# You will have X, y in your environment
load('spamTrain.Rda')
list2env(data,.GlobalEnv)
```

    ## <environment: R_GlobalEnv>

``` r
rm(data)

cat(sprintf('\nTraining Linear SVM (Spam Classification)\n'))
```

    ## 
    ## Training Linear SVM (Spam Classification)

``` r
cat(sprintf('(this may take 1 to 2 minutes) ...\n'))
```

    ## (this may take 1 to 2 minutes) ...

``` r
C <- 0.1
model <- svmTrain(X, y, C, linearKernel)
```

    ## 
    ## Training ......................................................................
    ## ...............................................................................
    ## ...............................................................................
    ## ....... Done!

``` r
p <- svmPredict(model, X)

cat(sprintf('Training Accuracy: %f\n', mean(p==y) * 100))
```

    ## Training Accuracy: 99.875000

Part 4: Test Spam Classification
--------------------------------

``` r
# Load the test dataset
# You will have Xtest, ytest in your environment
load('spamTest.Rda')
list2env(data,.GlobalEnv)
```

    ## <environment: R_GlobalEnv>

``` r
rm(data)

cat(sprintf('\nEvaluating the trained Linear SVM on a test set ...\n'))
```

    ## 
    ## Evaluating the trained Linear SVM on a test set ...

``` r
p <- svmPredict(model, Xtest)

cat(sprintf('Test Accuracy: %f\n', mean(p==ytest) * 100))
```

    ## Test Accuracy: 98.700000

Part 5: Top Predictors of Spam
------------------------------

``` r
# Sort the weights and obtin the vocabulary list
srt <- sort(model$w, decreasing = TRUE,index.return=TRUE)
weight <- srt$x
idx <- srt$ix
rm(srt)

vocabList <- getVocabList()

cat(sprintf('\nTop predictors of spam: \n'))
```

    ## 
    ## Top predictors of spam:

``` r
for (i in 1:15)
    cat(sprintf(' %-15s (%f) \n', vocabList[idx[i]], weight[i]))
```

    ##  our             (0.499153) 
    ##  click           (0.470095) 
    ##  remov           (0.421643) 
    ##  guarante        (0.381713) 
    ##  visit           (0.368689) 
    ##  basenumb        (0.345914) 
    ##  dollar          (0.331576) 
    ##  will            (0.267814) 
    ##  price           (0.263253) 
    ##  pleas           (0.260861) 
    ##  most            (0.259387) 
    ##  lo              (0.259144) 
    ##  nbsp            (0.252187) 
    ##  dollarnumb      (0.241453) 
    ##  hour            (0.237956)

``` r
cat(sprintf('\n\n'))
```

Part 6: Try Your Own Emails
---------------------------

``` r
# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
fName <- 'spamSample1.txt'

# Read and predict
file_contents <- readChar(fName,file.info(fName)$size)
```

    ## Warning in readChar(fName, file.info(fName)$size): 在non-UTF-8 MBCS语言环境
    ## 里只能读取字节

``` r
word_indices  <- processEmail(file_contents)
```

    ## 
    ## ----- Processed Email -----
    ## 
    ## do you want to make dollarnumb or more per week if you ar a motiv and qualifi individu i will person demonstr to you a system that will make you dollarnumb number per week or more thi i not mlm call our number hour pre record number to get the detail number number number i need peopl who want to make seriou monei make the call and get the fact invest number minut in yourself now number number number look forward to your call and i will introduc you to peopl like yourself who ar current make dollarnumb number plu per week number number number numberljgvnumb numberleannumberlrmsnumb numberwxhonumberqiytnumb numberrjuvnumberhqcfnumb numbereidbnumberdmtvlnumb
    ## 
    ## --------------------------

``` r
x             <- emailFeatures(word_indices)
p <- svmPredict(model, x)

cat(sprintf('\nProcessed %s\n\nSpam Classification: %d\n', fName, p))
```

    ## 
    ## Processed spamSample1.txt
    ## 
    ## Spam Classification: 1
