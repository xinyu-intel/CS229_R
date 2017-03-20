K-Means Clustering with R
================

Initialization
--------------

``` r
rm(list=ls())
sources <- c("computeCentroids.R",
             "findClosestCentroids.R","kMeansInitCentroids.R",
             "plotDataPoints.R","plotProgresskMeans.R","runkMeans.R")

for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}
```

    ## Loading  computeCentroids.R 
    ## Loading  findClosestCentroids.R 
    ## Loading  kMeansInitCentroids.R 
    ## Loading  plotDataPoints.R 
    ## Loading  plotProgresskMeans.R 
    ## Loading  runkMeans.R

Part 1: Find Closest Centroids
------------------------------

``` r
cat(sprintf('Finding closest centroids.\n\n'))
```

    ## Finding closest centroids.

``` r
# Load an example dataset that we will be using
load('ex7data2.Rda')
list2env(data,.GlobalEnv)
```

    ## <environment: R_GlobalEnv>

``` r
rm(data)

# Select an initial set of centroids
K <- 3; # 3 Centroids
initial_centroids <- matrix(c(3, 3, 6, 2, 8, 5),3,2,byrow = TRUE)

# Find the closest centroids for the examples using the
# initial_centroids
idx <- findClosestCentroids(X, initial_centroids)

cat(sprintf('Closest centroids for the first 3 examples: \n'))
```

    ## Closest centroids for the first 3 examples:

``` r
cat(sprintf(' %d', idx[1:3]))
```

    ##  1  3  2

``` r
cat(sprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n'))
```

    ## 
    ## (the closest centroids should be 1, 3, 2 respectively)

Part 2: Compute Means

``` r
cat(sprintf('\nComputing centroids means.\n\n'))
```

    ## 
    ## Computing centroids means.

``` r
#  Compute means based on the closest centroids found in the previous part.
centroids <- computeCentroids(X, idx, K)

cat(sprintf('Centroids computed after initial finding of closest centroids: \n'))
```

    ## Centroids computed after initial finding of closest centroids:

``` r
cat(sprintf(' %f %f \n' , centroids[,1],centroids[,2]))
```

    ##  2.428301 3.157924 
    ##   5.813503 2.633656 
    ##   7.119387 3.616684

``` r
cat(sprintf('\n(the centroids should be\n'))
```

    ## 
    ## (the centroids should be

``` r
cat(sprintf('   [ 2.428301 3.157924 ]\n'))
```

    ##    [ 2.428301 3.157924 ]

``` r
cat(sprintf('   [ 5.813503 2.633656 ]\n'))
```

    ##    [ 5.813503 2.633656 ]

``` r
cat(sprintf('   [ 7.119387 3.616684 ]\n\n'))
```

    ##    [ 7.119387 3.616684 ]

Part 3: K-Means Clustering

``` r
cat(sprintf('\nRunning K-Means clustering on example dataset.\n\n'))
```

    ## 
    ## Running K-Means clustering on example dataset.

``` r
# Load an example dataset
load("ex7data2.Rda")
list2env(data,.GlobalEnv)
```

    ## <environment: R_GlobalEnv>

``` r
rm(data)

# Settings for running K-Means
K <- 3
max_iters <- 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids <- matrix(c(3, 3, 6, 2, 8, 5),3,2,byrow = TRUE)

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
kMean <- runkMeans(X, initial_centroids, max_iters, FALSE)
```

    ## K-Means iteration 1/10...
    ## K-Means iteration 2/10...
    ## K-Means iteration 3/10...
    ## K-Means iteration 4/10...
    ## K-Means iteration 5/10...
    ## K-Means iteration 6/10...
    ## K-Means iteration 7/10...
    ## K-Means iteration 8/10...
    ## K-Means iteration 9/10...
    ## K-Means iteration 10/10...

``` r
centroids <- kMean$centriods
idx <- kMean$idx

cat(sprintf('\nK-Means Done.\n\n'))
```

    ## 
    ## K-Means Done.

Part 4: K-Means Clustering on Pixels

``` r
cat(sprintf('\nRunning K-Means clustering on pixels from an image.\n\n'))
```

    ## 
    ## Running K-Means clustering on pixels from an image.

``` r
#  Load an image of a bird
# library(png)
# A <- readPNG('bird_small.png')
#Instead load from Rda
load('bird_small.Rda')
list2env(data,.GlobalEnv)
```

    ## <environment: R_GlobalEnv>

``` r
rm(data)

A <- A / 255 # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size <- dim(A)

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X <- matrix(A, img_size[1] * img_size[2], 3)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K <- 16
max_iters <- 10

# When using K-Means, it is important the initialize the centroids
# randomly.
# You should complete the code in kMeansInitCentroids.R before proceeding
initial_centroids <- kMeansInitCentroids(X, K)


# Run K-Means
kMean <- runkMeans(X, initial_centroids, max_iters)
```

    ## K-Means iteration 1/10...
    ## K-Means iteration 2/10...
    ## K-Means iteration 3/10...
    ## K-Means iteration 4/10...
    ## K-Means iteration 5/10...
    ## K-Means iteration 6/10...
    ## K-Means iteration 7/10...
    ## K-Means iteration 8/10...
    ## K-Means iteration 9/10...
    ## K-Means iteration 10/10...

``` r
centroids <- kMean$centroids
idx <- kMean$idx
```

Part 5: Image Compression

``` r
cat(sprintf('\nApplying K-Means to compress an image.\n\n'))
```

    ## 
    ## Applying K-Means to compress an image.

``` r
# Find closest cluster members
idx <- findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx.

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by it's index in idx) to the centroid value
X_recovered <- centroids[idx,]

# Reshape the recovered image into proper dimensions
X_recovered <- array(X_recovered, img_size) #3 dimensional array

# Display the original image
op <- par(mfrow=c(1,2),mar=c(5, 4, 4, 2) + .1)

library(raster)
```

    ## Loading required package: sp

``` r
b <- brick(A)
plotRGB(b,stretch='lin',asp=2 ,axes=TRUE, main="Original")

# Display compressed image side by side
b <- brick(X_recovered)
plotRGB(b,stretch='lin',asp=2,axes=TRUE, main=sprintf('Compressed,\n with %d colors.', K))
```

![](ex6_K-Means_Clustering_with_R_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
par(op)
```
