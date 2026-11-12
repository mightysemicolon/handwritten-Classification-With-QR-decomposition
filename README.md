# Handwritten Digit Classification with QR and SVD

Classifying handwritten digits with only linear algebra, no neural network and no training in the usual sense. Each digit gets its own subspace, and a new image is assigned to the digit whose subspace it is closest to.

There are two notebooks. The first one uses the training images directly with QR, the second one first builds an SVD basis for each digit.

## Data

TinyMNIST: 28 x 28 grayscale images of digits 0 to 9. Every image is flattened to a vector of length 784.

The test set has 600 images, 60 for each digit, with labels in `test labels.csv`.

The data is in `TinyMNIST.rar`. Extract it next to the notebooks so that the folder `TinyMNIST/` exists.

## Method 1: least squares with QR (`Project-3-QR.ipynb`)

For each digit `d`, take 105 training images and put them as columns of a matrix `A_d` (784 x 105).

For a test image `b`, solve

```
min_x || A_d x - b ||
```

for all ten digits, using the QR decomposition of `A_d`. The residual tells how well `b` can be written as a combination of the training images of that digit. The digit with the smallest residual is the prediction.

Accuracy: **92.33%**

## Method 2: SVD basis (`Project-3-QR_SVD.ipynb`)

Here I use 400 training images per digit, so `A_d` is 784 x 400. Instead of using all the columns, I compute the SVD of `A_d` and keep only the first 20 left singular vectors. These 20 vectors are the main "directions" of that digit, and the rest is mostly individual handwriting differences and noise.

The classification step is the same as before, with the 784 x 20 basis in place of `A_d`.

Accuracy: **95.00%**

## Results

Accuracy for each digit (%):

| Digit | QR (105 images) | SVD (rank 20, 400 images) |
|---|---|---|
| 0 | 100.00 | 100.00 |
| 1 | 95.00 | 96.67 |
| 2 | 93.33 | 91.67 |
| 3 | 95.00 | 95.00 |
| 4 | 93.33 | 96.67 |
| 5 | 90.00 | 93.33 |
| 6 | 98.33 | 95.00 |
| 7 | 81.67 | 93.33 |
| 8 | 85.00 | 95.00 |
| 9 | 91.67 | 93.33 |
| **All** | **92.33** | **95.00** |

The biggest difference is for 7 and 8, which were the weak digits in the first method. With the SVD basis all digits are above 91%. It is also cheaper at test time, since each least squares problem has 20 unknowns instead of 105.

Both notebooks also plot, for each true digit, how the predictions are distributed, so you can see which digits get confused with each other.

## Requirements

numpy, pandas, matplotlib

## Reference

Lars Eldén, *Matrix Methods in Data Mining and Pattern Recognition*, SIAM. Chapter on classification of handwritten digits.
