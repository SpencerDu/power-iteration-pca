# power-iteration-pca

Finding the eigenvalues and eigenvectors of a symmetric matrix by power iteration, and PCA built on top of it, written by hand while following MIT OCW 18.06 (Gilbert Strang):

Power iteration: random vector v -> multiply by A -> scale it back to length 1 -> repeat until the eigenvalue stops changing -> v is the eigenvector with the biggest eigenvalue

PCA: data -> subtract the mean of each column -> covariance matrix -> its eigenvectors by power iteration, biggest eigenvalue first -> those are the directions the data varies along the most -> project the data onto them

numpy is only used for the arrays and their arithmetic and to check the answers, never as the eigenvalue solver.

## Why

To understand eigenvalues and eigenvectors better.

## How it works

- Power iteration: any vector is a mix of the eigenvectors of A. Multiplying by A scales each part by its own eigenvalue, so the part with the biggest eigenvalue (in size, ignoring the sign) grows the fastest. Repeat it and that part is all that is left. The vector is scaled back to length 1 every step so the numbers do not blow up.
- Eigenvalue (Rayleigh quotient): lambda = (v . Av) / (v . v). If v is an eigenvector then Av = lambda v, so this gives lambda.
- Stopping: when lambda changes by less than 1e-9 in one step, or after 1000 steps. It watches lambda and not v: with a negative eigenvalue v flips sign every step and never settles, but lambda does.
- Next eigenvalue (deflation): subtract lambda times v v^T from the matrix. That takes out the direction just found, so the second biggest eigenvalue is now the biggest and power iteration finds it next. This needs a symmetric matrix, where the eigenvectors are perpendicular.
- PCA: subtract the mean of each column to get Xc, then the covariance matrix is C = Xc^T Xc / (m - 1) for m data points. Its eigenvectors are the directions the data varies along the most, and each eigenvalue is the variance along that direction. Xc times the eigenvectors gives the data in the new axes.

## Results

- [[2, 1], [1, 2]] gives lambda = 3 and v along (1, 1).
- [[-3, 0], [0, 1]] gives lambda = -3, even though v flips sign every step.
- The covariance matrix matches np.cov, and the diagonal test matrix (2, 5, 3) gives 5, 3, 2 by deflation, the same as numpy.
- PCA on five points on the line y = 2x: one direction, along (1, 2), carries all of the variance, and the projected values are how far each point is along that line.
- The biggest eigenvalue matches numpy to 1e-6 on 981 out of 1000 random symmetric 5x5 matrices.
- On 200 random covariance matrices all 5 eigenvalues by deflation are within 4e-9 of numpy.
- PCA on 200 random data sets: the variances are within 4e-9 of numpy and the projected data within 5e-4 (up to sign).

## Known limitations

- The eigenvectors are less accurate than the eigenvalues. The loop stops when lambda stops changing, and lambda settles sooner than v (its error goes like the square of the error in v). So lambda is good to about 1e-9 but v only to about 1e-5. The script's is_eigenpair checks are set at 1e-5, right at that level, so they often print False even though the eigenvalue is right, nearly every run for Test 3.
- It is slow when the two biggest eigenvalues are close in size, and it does not say when it has failed. With the second at 0.5 of the biggest it takes about 16 steps, at 0.9 about 90, and at 0.99 most of its 1000 step limit. Closer than that it runs out of steps and returns what it has. The 19 misses above are all like this. Rarely it also stops too early, if the random start has almost none of the biggest eigenvector in it.
- If the two biggest are the same size with opposite signs, like 3 and -3, it returns a number anywhere between -3 and 3.
- In the last test the five points lie exactly on a line, so there is no second direction to find. What is left of the matrix is 0 apart from rounding, and power iteration on that returns a copy of the first direction or nan. The second column of projected in that test means nothing. It should be all zeros.

## Running it

Needs Python with numpy:

    pip install numpy
    python power_iteration.py

It prints the power iteration tests, the covariance matrix, the eigenvalues by deflation and the PCA result, with numpy's answer next to the ones that have a check. The start vector is random, so the signs of the eigenvectors and the random matrix in Test 3 change from run to run.
