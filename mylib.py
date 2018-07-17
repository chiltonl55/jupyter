# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:16:08 2017

@author: chilton
"""
import numpy as np
import sys

###
# code for algorithms in Sauer
###

# Chapter 1

def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

# Computes approximate solution of f(x)=0 using bisection method
# Input: function f, a,b such that f(a)*f(b)<0, tolerance tol
# Output: Approximate solution xc and number of iterations
def bisect(f, a, b, tol):
# evaluate f(x) at a and b
    fa = f(a)
    fb = f(b)
    n = 0
# check to ensure root is bracketed
    if sign(fa)*sign(fb) >= 0:
        print('f(a)f(b)<0 not satisfied!')
        quit()
# repeat loop until the interval is small enough
    while (b-a)/2. > tol:
# count number if iterations
        n = n + 1
# compute centerpoint of interval
        c = (a+b)/2.
# evaluate f at centerpoint
        fc = f(c)
        if fc == 0:			# c is a solution, done
            return c
# check if a and c bracket root, if yes, then b = c, otherwise a = c
        if sign(fc)*sign(fa) < 0:	# a and c make the new interval
            b = c
            fb = fc
        else:				# c and b make the new interval
            a = c
            fa = fc
# return root and number of iterations
    return [(a+b)/2., n]		# new midpoint is best estimate

# Computes approximate solution of f(x)=0 using Newton's method
# Input: f function, fp derivative of function, x0 intitial guess, k number of iterations
# Output: Approximate root xc
def newton(f, fp, x0, k):
    xc = x0
    for i in range(1, k):
        xc = xc - f(xc)/fp(xc)
    return xc


# Computes approximate solution of f(x)=0 using secant method
# Input: f function, x0 and x1 intitial guesses, k number of iterations
# Output: x2 Approximate root
def secant(f, x0, x1, k):
    for i in range(1, k):
        x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
        x0 = x1
        x1 = x2
    return x2

# Chapter 2

# factors the matrix a into a=lu where l is lower triangular and u is upper triangular
# input: square matrix a
# output: lower triangular matrix l and upper triangular triangular matrix u
def lu(a):
# make sure a is square
    [m, n] = a.shape
    if m != n:
        sys.exit("Matrix must be square")
    a = a.astype(float)

# allocate memory for l
    l = np.eye(n)

# for columns 0 thru n-2
    for k in np.arange(n-1):
        if a[k,k] == 0:
            sys.exit("Matrix must be square")

# in column k, create zeros in each row below the diagonal
# Compute multiplier for each row
        a[k+1:n,k] = a[k+1:n,k]/a[k,k]

# Update the remainder of the matrix
# np.newaxis forces result to be a 2D array
        a[k+1:n,k+1:n] = a[k+1:n,k+1:n] - a[k+1:n,k,np.newaxis]*a[k,k+1:n]        

# extract l and u from a
    l = l + np.tril(a,-1)
    u = np.triu(a)
    return (l,u)

# factors the matrix a into pa=lu where p is a permutation matrix,
# l is lower triangular and u is upper triangular
# input: square matrix a
# output: permutation matrix p, lower triangular matrix l and
# upper triangular triangular matrix u
def pa_lu(a):
# make sure a is square
    [m, n] = a.shape
    if m != n:
        sys.exit("Matrix must be square")
    a = a.astype(float)

# allocate memory for p, l
    p = np.eye(n)
    l = np.eye(n)

# for columns 0 thru n-2
    for k in np.arange(n-1):

# pivot if necessary (move row with max entry below diag to diag)
        mi = np.argmax(np.abs(a[k:(m), k]))
        mi = mi + k

# swap rows of both p and a
        if mi > k:
            p[[k,mi],:] = p[[mi,k],:]
            a[[k,mi],:] = a[[mi,k],:]

# in column k, create zeros in each row below the diagonal
# Compute multiplier for each row
        a[k+1:n,k] = a[k+1:n,k]/a[k,k]

# Update the remainder of the matrix
# np.newaxis forces result to be a 2D array
        a[k+1:n,k+1:n] = a[k+1:n,k+1:n] - a[k+1:n,k,np.newaxis]*a[k,k+1:n]        

# extract l and u from a
    l = l + np.tril(a,-1)
    u = np.triu(a)
    return (p,l,u)