---
layout: page
title:  "Fundamentals of Digital Image and Video Processing"
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


{:toc}

* 
{:toc}


<style>
table {
  border-collapse: collapse;
  border: 1px solid black;
  margin: 0 auto;
} 

th,td {
  border: 1px solid black;
  text-align: center;
  padding: 20px;
}

table.a {
  table-layout: auto;
  width: 180px;  
}

table.b {
  table-layout: fixed;
  width: 600px;  
}

table.c {
  table-layout: auto;
  width: 100%;  
}

table.d {
  table-layout: fixed;
  width: 100%;  
}
</style>


<p align="justify">
<a href="https://www.coursera.org/account/accomplishments/certificate/FQ4XNHK9HLBF"> My certificate.</a><br>
</p>


## 1. Introduction to Image and Video Processing
### 1.1 Analog vs. Digital Signals
#### 1.1.1 digital signals
### 1.2 Image and video signal
### 1.3 Electromagnetic spectrum
#### 1.3.1 Example EM
<p align="justify">
Microwave<br>
light from the sun
</p>


## 2. Signal and system
### 2.1 2D and 3D discrete signals
#### 2.1.1 Discrete unit impulse
#### 2.1.2 Separable signals
<p align="justify">
$$g(n_{1}, n_{2}) = f_{1}(n_{1}) * f_{2}(n_{2})$$
</p>

#### 2.1.3 Discrete unit step
### 2.2 Complex exponential signal
<p align="justify">
$$x(n_{1}, n_{2}) = e^{j\omega_{1}n_{1}} e^{j\omega_{2}n_{2}}$$
</p>

### 2.3 Linear shift invariant system
<p align="justify">
A linear shift-invariant system is fully characterizable by its impulse response.
</p>

### 2.4 2D convolution
<p align="justify">
$$y(n_{1}, n_{2}) = x(k_{1}, k_{2})**h(n_{1}, n_{2}) = \sum_{k_{1}=-\infty}^{\infty}\sum_{k_{2}=-\infty}^{\infty} x(k_{1}, k_{2}) h(n_{1}-k_{1}, n_{2}-k_{2})$$

$$B(i, j) = \sum_{m=0}\sum_{n=0}K(m, n)A(i-m, j-n)$$

$$S(n) = (f*g)[ n ] = \sum_{m=0}^{N-1} f(m)g(n-m)$$

N -- length of signal $f[ n ]$<br>
$S(n)$ -- result series length = len($f[ n ]$) + len($g[ n ]$) - 1<br><br>

Example $f[ n ] = [1, 2, 3]$ and $g[ n ] = [2, 3, 1]$
$$S(0) = \sum_{m=0}^{2} f(m)g(0-m) = f(0)g(0) + f(1)g(-1) + f(2)g(-2) = 1\cdot 2 + 2\cdot 0 + 3\cdot 0 = 2$$

$$S(1) = \sum_{m=0}^{2} f(m)g(1-m) = f(0)g(1) + f(1)g(0) + f(2)g(-1) = 1\cdot 3 + 2\cdot 2 + 3\cdot 0 = 7$$

$$S(2) = \sum_{m=0}^{2} f(m)g(2-m) = f(0)g(2) + f(1)g(1)+ f(2)g(0) = 1\cdot 1 + 2\cdot 3 + 3\cdot 2 = 13$$

$$S(3) = \sum_{m=0}^{2} f(m)g(3-m) = f(0)g(3) + f(1)g(2) + f(2)g(1) = 1\cdot 0 + 2\cdot 1 + 3\cdot 3 = 11$$

$$S(4) = \sum_{m=0}^{2} f(m)g(4-m) = f(0)g(4) + f(1)g(3) + f(2)g(2) = 1\cdot 0 + 2\cdot 0 + 3\cdot 1 = 3$$

Reflection, shift, sum
</p>

### 2.5 Filtering in spatial domian
#### 2.5.1 Boundary effect
#### 2.5.2 Spatial filtering
#### 2.5.3 Noise reduction
#### 2.5.4 High pass filter
<p align="justify">
An image is sharpened when contrast is enhanced between adjoining areas with little variation in brightness or darkness. A high pass filter tends to retain the high frequency information within an image while reducing the low frequency information. The kernel of the high pass filter is designed to increase the brightness of the center pixel relative to neighboring pixels. The kernel array usually contains a single positive value at its center, which is completely surrounded by negative values. The following array is an example of a 3 by 3 kernel for a high pass filter:
$$
\frac{1}{9}
\begin{vmatrix}
-1 & -1 & -1\\
-1 & 8 & -1\\
-1 & -1 & -1
\end{vmatrix}
$$
</p>

#### 2.5.5 Low pass filter
<p align="justify">
An image is smoothed by decreasing the disparity between pixel values by averaging nearby pixels. Using a low pass filter tends to retain the low frequency information within an image while reducing the high frequency information. An example of a low pass filter is an array of ones divided by the number of elements within the kernel, such as the following 3×3 kernel
$$
\frac{1}{9}
\begin{vmatrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1
\end{vmatrix}
$$
</p>

## 3. Fourier transform and sampling
### 3.1 2D Fourier transform
<p align="justify">
$$x(n_{1}, n_{2}) = e^{\omega_{1}^{'}n_{1} + \omega_{2}^{'}n_{2}} \rightarrow h(n_{1}, n_{2}) \rightarrow y(n_{1}, n_{2})$$

$$y(n_{1}, n_{2}) = x(n_{1}, n_{2})**h(n_{1}, n_{2}) = \sum_{k_{1}=-\infty}^{\infty}\sum_{k_{2}=-\infty}^{\infty} e^{j\omega_{1}^{'}(n_{1}-k_{1})} e^{j\omega_{2}^{'}(n_{2}-k_{2})} h(k_{1}, k_{2})$$

$$= e^{j\omega_{1}^{'}n_{1}} e^{j\omega_{2}^{'}n_{2}} \sum_{k_{1}=-\infty}^{\infty}\sum_{k_{2}=-\infty}^{\infty} h(k_{1}, k_{2}) e^{-j\omega_{1}^{'}k_{1}} e^{-j\omega_{2}^{'}k_{2}}$$

Frequency response
$$H(\omega_{1}^{'}, \omega_{2}^{'}) = \sum_{k_{1}=-\infty}^{\infty}\sum_{k_{2}=-\infty}^{\infty} h(k_{1}, k_{2}) e^{-j\omega_{1}^{'}k_{1}} e^{-j\omega_{2}^{'}k_{2}}$$

$$X(\omega_{1}, \omega_{2}) = \sum_{n_{1}=-\infty}^{\infty}\sum_{n_{2}=-\infty}^{\infty} x(n_{1}, n_{2}) e^{-j\omega_{1}n_{1}} e^{-j\omega_{2}n_{2}}$$

$$x(n_{1}, n_{2}) = \frac{1}{4\pi^{2}} \int_{-\pi}^{\pi}\int_{-\pi}^{\pi} X(\omega_{1}, \omega_{2}) e^{j\omega_{1}n_{1}} e^{j\omega_{2}n_{2}} d\omega_{1}d\omega_{2}$$

Property:
$$X(\omega_{1}, \omega_{2}) = X(\omega_{1}+2\pi, \omega_{2}+2\pi)$$

$$x(n_{1}-m_{1}, n_{2}-m_{2}) \leftrightarrow e^{-j\omega_{1}n_{1}} e^{-j\omega_{2}n_{2}} X(\omega_{1}, \omega_{2})$$

$$x(n_{1}, n_{2})e^{j\theta_{1}n_{1}+j\theta_{2}n_{2}} \leftrightarrow X(\omega_{1}-\theta_{1}, \omega_{2}-\theta_{2})$$

For x(n_{1}, n_{2}) real
$$|X(\omega_{1}, \omega_{2})| = |X(-\omega_{1}, -\omega_{2})|$$

$$arg X(\omega_{1}, \omega_{2}) = -argX(-\omega_{1}, -\omega_{2})$$

$$\int_{n_{1}}\int_{n_{2}} |x(n_{1}, n_{2})|^{2} = \frac{1}{4\pi^{2}} \int_{-\pi}^{\pi}\int_{-\pi}^{\pi} |x(\omega_{1}, \omega_{2})|^{2} d\omega_{1}d\omega_{2}$$

Example
$$
h(n_{1}, n_{2}) = 
\begin{vmatrix}
h(-1, 1) & h(0, 1) & h(1, 1)\\
h(-1, 0) & h(0, 0) & h(0, 1)\\
h(-1, -1) & h(0, -1) & h(1, -1)
\end{vmatrix} = 
\begin{vmatrix}
0.075 & 0.124 & 0.075\\
0.124 & 0.204 & 0.124\\
0.075 & 0.124 & 0.075
\end{vmatrix}
$$

$$H(\omega_{1}, \omega_{2}) = 0.204 + 0.124 \cdot 2 \cdot cos\omega_{1} + 0.124\cdot 2 \cdot cos\omega_{2} + 0.075 \cdot cos(\omega_{1}+\omega_{2}) + 0.075 \cdot 2 \cdot cos(\omega_{1} - \omega_{2})$$

$$
h(n_{1}, n_{2}) = 
\begin{vmatrix}
h(-1, 1) & h(0, 1) & h(1, 1)\\
h(-1, 0) & h(0, 0) & h(0, 1)\\
h(-1, -1) & h(0, -1) & h(1, -1)
\end{vmatrix} = 
\begin{vmatrix}
-1 & -1 & -1\\
-1 & 9 & -1\\
-1 & -1 & -1
\end{vmatrix}
$$

$$H(\omega_{1}, \omega_{2}) = 9 - 2 \cdot cos\omega_{1} - 2 \cdot cos\omega_{2} - 2 \cdot cos(\omega_{1}+\omega_{2}) - 2 \cdot cos(\omega_{1} - \omega_{2})$$
</p>

### 3.2 Sampling
#### 3.2.1 2D sampling
<p align="justify">
$$X(\Omega_{1}T_{1}, \Omega_{2}T_{2}) = \frac{1}{T_{1}T_{2}} \sum_{k_{1}=-\infty}^{\infty}\sum_{k_{2}=-\infty}^{\infty} X_{\alpha}(\Omega_{1}-\frac{2\pi}{k_{1}}k_{1}, \Omega_{2}-\frac{2\pi}{k_{2}}k_{2})$$
</p>

#### 3.2.2 Critically sampled
#### 3.2.3 Over-sampled
#### 3.2.4 Under-sampled
#### 3.2.5 2D Nyquist theorem

### 3.3 Discrete Fourier transform
#### 3.3.1 DFT
#### 3.3.2 FFT
### 3.4 Filtering in the Frequency Domain
<p align="justify">
The ROS of the signal resulting from the convolution of two nonzero signals is at least as large as each of the ROS's of the two signals being convolved.
</p>

### 3.5 Change of sampling rate
#### 3.5.1 Down sampling
<p align="justify">
In order to avoid aliasing when down-sampling an image, low-pass filtering needs to be done prior to the down-sampling
</p>

#### 3.5.2 Up sampling
