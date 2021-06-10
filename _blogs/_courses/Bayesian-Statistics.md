---
layout: page
title:  "Bayesian Statistics"
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
<a href="https://www.coursera.org/account/accomplishments/certificate/7HXV69TQH9K8"> My certificate 1.</a><br>
</p>


## 1. Bayesian Statistics: From Concept to Data Analysis
<p align="justify">
There are two main philosophies of probability and statistics, Bayesian and Frequentist. In particular, the Bayesian approach is better for dealing with uncertainty. Both in terms in being able to quantify uncertainty, and also for being able to combine uncertainties in a coherent manner.<br><br>
</p>

### 1.1 Probability
#### 1.1.1 Background
<p align="justify">
<b>Rules of Probability</b><br>
Probabilities are defined for events. For example, rolling a fair six-sided die, This event has probability $\frac{1}{6}$.<br>
$$P(X = 4) = \frac{1}{6}$$

Probabilities must be between zero and one for any A.<br>
$$0 \leq P(A) \leq 1$$

Probabilities add to one<br>
$$\sum_{i=1}^{6} P(X = i) = 1$$

If A and B are two events, the probability that A or B happens (this is an inclusive or, meaning that either A, or B, or both happen) is the probability of the union of the events:<br>
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

The complement of an event $A^{c}$, means that the event does not happen.<br>
$$P(A^{c}) = 1 - P(A)$$

For i = 1, 2, ..., m are mutually exclusive (only one can happen), then<br>
$$P(\cup_{i=1}^{m} A_{i}) = \sum_{i=1}^{m} P(A_{i})$$

<b>Odds</b><br>
Probabilities can be re-expressed in terms of odds.<br>
$$O(A) = \frac{P(A)}{P(A^{c})} = \frac{P(A)}{1 - P(A)}$$

<b>Expectation</b><br>
The expected value of a random variable X is a weighted average of values X can take, with weights given by the probabilities of those values.<br><br>

If X can take on only a finite number of values, say $x_{1}$, $x_{2}$, ..., $x_{n}$<br>
$$E(X) = \sum_{i=1}^{n} x_{i} \cdot P(X = x_{i})$$

Expectation for a continuous random variable or a discrete random variable<br>
$$E[X]=\int_{-\infty }^{\infty}xf(x)dx=\sum xf(x)$$

Expectation for some function of random variable x<br>
$$E[g(x)]=\int_{-\infty }^{\infty}g(x)f(x)dx$$

Expectation for a constant multiplying a random variable<br>
$$E[cX]=xE[X]$$

Expectation for a sum of two random variables<br>
$$E[X+Y]=E[X]+E[Y]$$

Expectation for a multiplication of two independent random variables<br>
$$E[XY]=E[X]E[Y], \quad \text{if} \quad X\perp Y$$

<b>Variance</b><br>
Two random variables are independent, the variance of their sum is the sum of their variances.<br>
$$Var[X+Y]=Var[X]+Var[Y]$$

Two random variables are dependent<br>
$$Var[X+Y]=Var[X]+Var[Y]+2Cov[X, Y]$$
$$Cov[X, Y]=E[(X-E[X])(Y-E[Y])]$$
$$Var[X]=E[(X-E[X])^{2}]=E[X^{2}]-(E[X])^{2}$$
$$
\begin{aligned}
Var[X+Y] & = E[(X+Y-E[X+Y])^{2}] \\
& = E[(X+Y)^{2}]-(E[X+Y])^{2} \\
& = E[X^{2}+2XY+Y_{2}]-(E[X]+E[Y])^{2} \\
& = E[X^{2}]-(E[X])^{2}+2E[XY]-2E[X]E[Y]+E[Y^{2}]-(E[Y])^{2}
\end{aligned}
$$

Therefore<br>
$$Cov[X, Y]=E[XY]-E[X]E[Y]$$<br>
</p>

#### 1.1.2 Classical and frequentist probability
<p align="justify">
One of the ways to deal with uncertainty is probabilities. There are three different frameworks under which we can define probabilities: <b>Classical framework</b>, <b>Frequentist framework</b> and <b>Bayesian framework</b>.<br><br>

Under the Classical framework, outcomes that are equally likely have equal probabilities. So in the case of rolling a fair die, there are six possible outcomes, they're all equally likely. So the probability of rolling a four, on a fair six sided die, is just one in six.<br>
$$P(X=4) = \frac{1}{6}$$

Frequentist definition, requires us to have a hypothetical infinite sequence of events, and then we look at the relevant frequency, in that hypothetical infinite sequence. In the case of rolling a die, a fair six sided die, we can think about rolling the die an infinite number of times. If it's a fair die and we roll infinite number of times then one sixth of the time, we'll get a four, showing up. And so we can continue to define the probability of rolling four in a six sided die as one in six.<br><br>

<b>The following argument best demonstrates reasoning from which paradigm?</b> On a multiple choice test, you do not know the answer to a question with three alternatives. You randomly select option (a), supposing you have a 1/3 chance of being correct.<br>
A. Classical<br>
B. Frequentist<br>
C. Bayesian<br>
<b>Answer</b>: A.<br><br>

<b>The following argument best demonstrates reasoning from which paradigm?</b> You survey 100 dental patients and find that 43 have received a filling within the last year. You conclude that the probability of needing a filling in any given year is .43.<br>
A. Classical<br>
B. Frequentist<br>
C. Bayesian<br>
<b>Answer</b>: B.<br><br>
</p>

#### 1.1.3 Bayesian probability and coherence
<p align="justify">
Bayesian perspective is one of personal perspective. Our probability represents your own perspective, it's our measure of uncertainty, and it takes into account what you know about a particular problem. But in this case, it's our personal one, and so what we see, may be different than what somebody else believes. For example we want to ask, is this is a fair die? If we have different information than somebody else then our probability of the die is fair may be different than that persons probability. So inherently a subjective approach to probability, but it can work well in a mathematically rigorous foundation, and it leads to much more intuitive results in many cases than the Frequentist approach.<br><br>

<b>Q</b>: Consider the following game: You flip a fair coin. If the result is “heads,” you win 3. If the result is “tails,” you lose 4. <b>Is this a fair game?</b><br>
A. Yes<br>
B. No<br>
<b>Answer</b>: B.<br><br>

expected return<br>
$$E[\text{win or lose money}] = 3 \times 0.5 + (-4) \times 0.5 = -0.5$$

Probabilities must follow all the standard rules of probability. If we don't follow all the rules for probability, then we can be incoherent, which leads to a case for someone to construct a series of bets where we are guaranteed to lose money.<br><br>
</p>

#### 1.1.4 Quiz
<p align="justify">
<b>1.</b><br>
If you randomly guess on this question, you have a .25 probability of being correct. <b>Which probabilistic paradigm from Lesson 1 does this argument best demonstrate?</b><br>
A. Classical<br>
B. Frequentist<br>
C. Bayesian<br>
D. None of the above<br>
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
On a multiple choice test, you do not know the answer to a question with three alternatives. One of the options, however, contains a keyword which the professor used disproportionately often during lecture. Rather than randomly guessing, you select the option containing the keyword, supposing you have a better than 1/3 chance of being correct. <b>Which probabilistic paradigm from Lesson 1 does this argument best demonstrate?</b><br>
A. Classical<br>
B. Frequentist<br>
C. Bayesian<br>
<b>Answer</b>: C.<br><br>

<b>3.</b><br>
On average, one in three students at your school participates in extracurricular activities. You conclude that the probability that a randomly selected student from your school participates is 1/3. <b>Which probabilistic paradigm from Lesson 1 does this argument best demonstrate?</b><br>
A. Classical<br>
B. Frequentist<br>
C. Bayesian<br>
<b>Answer</b>: B.<br><br>

<b>4. For Questions 4-6, consider the following scenario:</b><br>
Your friend offers a bet that she can beat you in a game of chess. If you win, she owes you 5, but if she wins, you owe her 3.<br><br>

Suppose she is 100% confident that she will beat you. <b>What is her expected return for this game?</b> (Report your answer without the $ symbol.)<br>
<b>Answer</b>: 3.<br><br>

<b>5.</b><br>
Suppose she is only 50% confident that she will beat you (her personal probability of winning is p=0.5). <b>What is her expected return now?</b> (Report your answer without the $ symbol.)<br>
<b>Answer</b>: -1.<br><br>

<b>6.</b><br>
Now assuming your friend will only agree to fair bets (expected return of 0), <b>find her personal probability that she will win</b>. Report your answer as a simplified fraction. Hint: Use the expected return of her proposed bet.<br>
<b>Answer</b>: $\frac{5}{8}$.<br><br>

<b>7. For Questions 7-8, consider the following "Dutch book" scenario:</b><br>
Suppose your friend offers a pair of bets:<br>
(i) if it rains or is overcast tomorrow, you pay him 4, otherwise he pays you 6;<br>
(ii) if it is sunny you pay him 5, otherwise he pays you 5.<br><br>

Suppose rain, overcast, and sunny are the only events in consideration. If you make both bets simultaneously, this is called a "Dutch book," as you are guaranteed to win money. How much do you win regardless of the outcome? (Report your answer without the $ symbol.)<br>
<b>Answer</b>: 1.<br>
If rain or overcast, 5-4; if sunny, 6-5.<br><br>

<b>8.</b><br>
Apparently your friend doesn't understand the laws of probability. Let's examine the bets he offered.<br>
For bet (i) to be fair, his probability that it rains or is overcast must be .6 (you can verify this by calculating his expected return and setting it equal to $0).<br>
For bet (ii) to be fair, his probability that it will be sunny must be .5.<br>
This results in a "Dutch book" because your friend's probabilities are not coherent. They do not add up to 1. What do they add up to?<br>
<b>Answer</b>: 0.6+0.5 = 1.1.<br><br>
</p>

### 1.2 Bayes' theorem
#### 1.2.1 Conditional probability
<p align="justify">
$$P(A \mid B)=\frac{P(A \bigcap B)}{P(B)}$$

For example, we have a table of student number.<br>
<table class="c">
  <tr><th></th><th>Female</th><th>Not female</th><th>Total</th></tr>
  <tr><td>CS major</td><td>4</td><td>8</td><td>12</td></tr>
  <tr><td>Not CS major</td><td>5</td><td>13</td><td>18</td></tr>
  <tr><td>Total</td><td>9</td><td>21</td><td>30</td></tr>
</table><br><br>
</p>
<p align="justify">
Then we have a proportional distribution.<br>
<table class="c">
  <tr><th></th><th>Female</th><th>Not female</th><th>Total</th></tr>
  <tr><td>CS major</td><td>2/15</td><td>4/15</td><td>2/5</td></tr>
  <tr><td>Not CS major</td><td>1/6</td><td>13/30</td><td>3/5</td></tr>
  <tr><td>Total</td><td>3/10</td><td>7/10</td><td>1</td></tr>
</table><br><br>
</p>
<p align="justify">
If we know a student is in computer science major, what is a probability that this sutudent is female<br>
$$P(\text{Female} \mid \text{CS})=\frac{P(\text{Female} \bigcap \text{CS})}{P(\text{CS})}=\frac{2/15}{6/15} = \frac{1}{3}$$

If A and B are independent<br>
$$P(A \mid B)=P(A)$$<br>
</p>

#### 1.2.2 Bayesian formula
<p align="justify">
$$P(A \mid B)=\frac{P(A \bigcap B)}{P(B)}=\frac{P(B \mid A)P(A)}{P(B \mid A)P(A) + P(B \mid A')P(A')}$$
Where A' is a complement of A.<br><br>

This equation is usually helpful, because we can transform a question of P(A | B) into another question P(B | A).<br><br>

We take HIV for instance, a person with HIV has a probability of P(+ | HIV) = 0.977 to be detected positive, and a person without HIV has P(- | !HIV) = 0.926 to be negative and there is P(HIV) = 0.0026 person with HIV. Now, what is a probability that a person detected as potive has HIV?<br>
$$P(\text{HIV} \mid +) = \frac{P(\text{HIV} \bigcap +)}{P(+)} = \frac{P(+ \mid \text{HIV})P(\text{HIV})}{P(+ \mid \text{HIV})P(\text{HIV})+P(+ \mid !\text{HIV})P(!\text{HIV})}$$

Here is a relation like P(- | !HIV) = 1-P(+ | !HIV). Then, we have a result<br>
$$P(\text{HIV} \mid +)=\frac{0.977 \times 0.0026}{0.977 \times 0.0026 + (1-0.926) \times (1-0.0026)} = 0.033$$<br>
</p>

#### 1.2.3 Supplementary material for Lesson 2
<p align="justify">
The simple form of Bayes Theorem involves two discrete events:<br>
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{P(B \mid A) P(A)}{P(B \mid A) P(A) + P(B \mid A^{c}) P(A^{c})}$$

When there are three possible outcomes $A_{1}$, $A_{2}$, and $A_{3}$ such that exactly one of these must happen, then Bayes Theorem expands to:<br>
$$P(A_{1} \mid B) = \frac{P(B \mid A_{1}) P(A_{1})}{P(B \mid A_{1}) P(A_{1}) + P(B \mid A_{2}) P(A_{2}) + P(B \mid A_{3}) P(A_{3})}$$

If the events $A_{1}$, ..., $A_{m}$ form a partition of the space (exactly one of the $A_{i}$’s must occur, i.e., the $A_{i}$’s are mutually exclusive and $\sum_{i=1}^{m} P(A_{i})$ = 1), then we can write Bayes Theorem as:<br>
$$P(A_{1} \mid B) = \frac{P(B \mid A_{1})P(A_{1})}{\sum_{i=1}^{m} P(B \mid A_{i}) P(A_{i})}$$<br>
</p>

#### 1.2.4 Quiz
<p align="justify">
<b>1.</b><br>
For Questions 1-4, refer to the following table regarding passengers of the famous Titanic, which tragically sank on its maiden voyage in 1912. The table organizes passenger/crew counts according to the class of their ticket and whether they survived.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_2_4_1.png"/></center>
</p>
<p align="justify">
If we randomly select a person's name from the complete list of passengers and crew, <b>what is the probability that this person travelled in 1st class?</b> Round your answer to two decimal places.<br>
<b>Answer</b>: (203 + 122) / (203 + 122 + 118 + 167 + 178 + 528 + 212 + 673) = 0.15.<br><br>

<b>2.</b><br>
<b>What is the probability that a (randomly selected) person survived?</b> Round your answer to two decimal places.<br>
<b>Answer</b>: (203 + 118 + 178 + 212) / (203 + 122 + 118 + 167 + 178 + 528 + 212 + 673) = 0.32.<br><br>

<b>3.</b><br>
What is the probability that a (randomly selected) person survived, given that they were in 1st class? Round your answer to two decimal places.<br><br>

<b>Answer</b>: (203) / (203 + 122) = 0.62.<br><br>

<b>4.</b><br>
True/False: The events concerning class and survival are statistically independent.<br>
A. True<br>
B. False<br>
<b>Answer</b>: B.<br>
$$P(\text{survived} \mid \text{1st class}) \neq P(\text{survived})$$

<b>5.</b><br>
For Questions 5-9, consider the following scenario:<br><br>

You have three bags, labeled A, B, and C. Bag A contains two red marbles and three blue marbles. Bag B contains five red marbles and one blue marble. Bag C contains three red marbles only.<br><br>

If you select from bag B, <b>what is the probability that you will draw a red marble?</b> Express the exact answer as a simplified fraction.<br>
<b>Answer</b>: $\frac{5}{6}$.<br><br>

<b>6.</b><br>
If you randomly select one of the three bags with equal probability (so that P(A) = P(B) = P(C) = 1/3) and then randomly draw a marble from that bag, <b>what is the probability that the marble will be blue?</b> Round your answer to two decimal places.<br>
<b>Answer</b>: 0.26.<br><br>

<b>7.</b><br>
Suppose a bag is randomly selected (again, with equal probability), but you do not know which it is. You randomly draw a marble and observe that it is blue. <b>What is the probability that the bag you selected this marble from is A?</b> That is, find P(A ∣ blue). Round your answer to two decimal places.<br>
<b>Answer</b>: 0.78.<br><br>

<b>8.</b><br>
Suppose a bag is randomly selected (again, with equal probability), but you do not know which it is. You randomly draw a marble and observe that it is blue. <b>What is the probability that the bag you selected from is C?</b> That is, find P(C | blue). Round your answer to two decimal places.<br>
<b>Answer</b>: 0.<br><br>

<b>9.</b><br>
Suppose a bag is randomly selected (again, with equal probability), but you do not know which it is. You randomly draw a marble and observe that it is red. <b>What is the probability that the bag you selected from is C?</b> That is, find P(C | red). Round your answer to two decimal places.<br>
<b>Answer</b>: 0.45.<br><br>
</p>

### 1.3 Review of distributions
<p align="justify">
When using random variable notation, <b>X</b> denotes a random variable while <b>x</b> denotes a value of a random variable.<br>
</p>

#### 1.3.1 Bernoulli and binomial distributions
<p align="justify">
<b>Bernoulli distribution</b><br>
$$X \sim B(p)$$
$$P(X = 1) = p, P(X = 0) = 1-p$$
$$f(X=x \mid p)=f(x \mid p)=p^{x}(1-p)^{1-x}$$
$$E[x]=\sum x P(X = x) = 1 \times p+0\times (1-p)=p$$
$$Var(X) = p(1-p)$$

Binomial is just the sum of the N independent Bernoullis<br>
$$X \sim Bin(n, p)$$
$$P(X = x \mid p) = f(x \mid p) = \binom{n}{x}p^{x}(1-p)^{n-x}, x\in (0, 1, ..., n)$$
$$E[X] = np$$
$$Var[X] = np(1-p)$$

<b>Q</b>: <b>Which of the following scenarios could we appropriately model using a binomial random variable?</b><br>
A. The number of failed lightbulbs in a batch of 5000 after 100 hours in service<br>
B. The hours of service until all 5000 light bulbs fail<br>
C. The expected lifetime of a lightbulb<br>
D. The probability of a light bulb failure before 100 hours in service<br>
<b>Answer</b>: A.<br><br>
</p>

#### 1.3.2 Quiz
<p align="justify">
<b>1.</b><br>
When using random variable notation, big X denotes ________.<br>
A. a random variable<br>
B. a conditional probability<br>
C. distributed as<br>
D. a realization of a random variable<br>
E. the expectation of a random variable<br>
F. approximately equal to<br>
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
When using random variable notation, little x denotes ________.<br>
A. a random variable<br>
B. a conditional probability<br>
C. distributed as<br>
D. a realization of a random variable<br>
E. the expectation of a random variable<br>
F. approximately equal to<br>
<b>Answer</b>: D.<br><br>

<b>3.</b><br>
When using random variable notation, X ~ denotes ________.<br>
A. a random variable<br>
B. a conditional probability<br>
C. distributed as<br>
D. a realization of a random variable<br>
E. the expectation of a random variable<br>
F. approximately equal to<br>
<b>Answer</b>: C.<br><br>

<b>4.</b><br>
What is the value of f(x) when x = 3?<br>
$$f(x) = -5 I_{x > 2}(x) + x I_{x < -1}(x)$$

<b>Answer</b>: -5.<br><br>

<b>5.</b><br>
What is the value of f(x) when x = 0?<br>
$$f(x) = -5 I_{x > 2}(x) + x I_{x < -1}(x)$$

<b>Answer</b>: 0.<br><br>

<b>6.</b><br>
Which of the following scenarios could we appropriately model using a Bernoulli random variable?<br>
A. Predicting whether your hockey team wins its next game (tie counts as a loss)<br>
B. Predicting the weight of a typical hockey player<br>
C. Predicting the number of goals scored in a hockey match<br>
D. Predicting the number of wins in a series of three games against a single opponent (ties count as losses)<br>
<b>Answer</b>: A.<br><br>

<b>7.</b><br>
Calculate the expected value of the following random variable: X takes on values {0, 1, 2, 3} with corresponding probabilities {0.5, 0.2, 0.2, 0.1}. Round your answer to one decimal place.<br>
<b>Answer</b>: 0.9.<br><br>

<b>8.</b><br>
Which of the following scenarios could we appropriately model using a binomial random variable (with n > 1)?<br>
A. Predicting the number of wins in a series of three games against a single opponent (ties count as losses)<br>
B. Predicting the number of goals scored in a hockey match<br>
C. Predicting the weight of a typical hockey player<br>
D. Predicting whether your hockey team wins its next game (tie counts as a loss)<br>
<b>Answer</b>: A.<br><br>

<b>9.</b><br>
Suppose X $\sim$ Binomial(3, 0.2). Calculate P(X = 0). Round your answer to two decimal places.<br>
<b>Answer</b>: 0.51.<br><br>

<b>10.</b><br>
Suppose X $\sim$ Binomial(3, 0.2). Calculate P(X $\leq$ 2). Round your answer to two decimal places.<br>
<b>Answer</b>: 0.99.<br><br>
</p>

#### 1.3.3 Uniform distribution
<p align="justify">
$$X \sim U[0, 1]$$
$$
f(x) =
\begin{cases}
  1, \quad x\in[0, 1] \\
  0, \quad \text{otherwise}
\end{cases}
$$
$$P(0 < x < 0.5) = \int_{0}^{0.5} f(x) dx = \int_{0}^{0.5} dx = 0.5$$

Becasue there are infinite number of possible outcomes, so the probability for any value in [0, 1] is 0.<br>
$$P(x=0.5)=0$$

There is another form<br>
$$X \sim [\theta_{1},\theta_{2} ]$$
$$f(x \mid \theta_{1}, \theta_{2}) = \frac{1}{\theta_{2}-\theta_{1}} \mathbb{I}_{\theta_{1} \leqslant x \leqslant \theta_{2}}$$
$$E[X] = \int_{\theta_{1}}^{\theta_{2}} x f(x) dx = \frac{1}{\theta_{2}-\theta_{1}} \int_{\theta_{1}}^{\theta_{2}} x dx = \frac{\theta_{1} + \theta_{2}}{2}$$

If X $\sim$ Uniform(0, 1), then the PDF of X is<br>
$$f(x) = I_{0 \leq x \leq 1}(x)$$

<b>Q</b>: Which of the following expressions could we use to calculate $E(X^{2})$?<br>
$$
\begin{aligned}
& \text{A.} \quad \int_{-\infty}^{\infty} (I_{0 \leq x \leq 1}(x))^{2} dx \\
\\
& \text{B.} \quad \int_{-\infty}^{\infty} x^{2} dx \\
\\
& \text{C.} \quad \int_{-\infty}^{\infty} I_{0 \leq x^{2} \leq 1}(x) dx \\
\\
& \text{D.} \quad \int_{0}^{1} x^{2} dx
\end{aligned}
$$

<b>Answer</b>: D.<br><br>
</p>

#### 1.3.4 Exponential and normal distribution
<p align="justify">
The exponential distribution is for a continuous random variable, e.g. the lifetime in hours of a particular lightbulb<br>
$$X \sim e^{\lambda}$$
$$f(x \mid \lambda) = \lambda e^{-\lambda x}, x \geq 0$$
$$E[X] = \frac{1}{\lambda}$$
$$Var[X] = \frac{1}{\lambda^{2}}$$

The normal distribution<br>
$$X \sim N(\mu, \sigma^{2})$$
$$f(x \mid \mu, \sigma^{2}) = \frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}}$$
$$E[X] = \mu$$
$$Var[X] = \sigma^{2}$$

<b>Q</b>: If X $\sim$ N($\mu$, $\sigma^{2}$), <b>what are the possible values X can take?</b><br>
$$
\begin{aligned}
& \text{A.} \quad -\infty < X < \infty \\
\\
& \text{B.} \quad 0 < X < \infty \\
\\
& \text{C.} \quad -1.96 < X < 1.96 \\
\\
& \text{D.} \quad -\mu < X < \mu
\end{aligned}
$$

<b>Answer</b>: A.<br><br>
</p>

#### 1.3.5 Quiz
<p align="justify">
<b>1.</b><br>
If continuous random variable X has probability density function (PDF) f(x), <b>what is the interpretation of the following integral?</b><br>
$$\int_{-2}^{5} f(x) dx$$<br>
A. $P(X \geq -2 \cap X \leq 5)$<br>
B. $P(X \leq -2 \cap X \leq 5)$<br>
C. $P(X \geq -2 \cup X \leq 5)$<br>
D. $P(X \leq -2 \cap X \geq 5)$<br>
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
If X $\sim$ Uniform(0, 1), then <b>what is the value of P(-3 < X < 0.2)</b><br>
<b>Answer</b>: 0.2.<br><br>

<b>3.</b><br>
If X $\sim$ Exponential(5), find the expected value E(X). (Round your answer to one decimal place.)<br>
<b>Answer</b>: 0.2.<br><br>

<b>4.</b><br>
Which of the following scenarios could we most appropriately model using an exponentially distributed random variable?<br>
A. The probability of a light bulb failure before 100 hours in service<br>
B. The lifetime in hours of a particular lightbulb<br>
C. The hours of service until all light bulbs in a batch of 5000 fail<br>
D. The number of failed lightbulbs in a batch of 5000 after 100 hours in service<br>
<b>Answer</b>: B.<br><br>

<b>5.</b><br>
If X $\sim$ Uniform(2, 6), <b>which of the following is the PDF of X?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_3_5_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: D.<br><br>

<b>6.</b><br>
If X $\sim$ Uniform(2, 6). <b>what is P(2 < X < 3)?</b> Round your answer to two decimal places.<br>
<b>Answer</b>: 0.25.<br><br>

<b>7.</b><br>
If X $\sim$ N(0, 1), <b>which of the following is the PDF of X?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_3_5_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: E.<br><br>

<b>8.</b><br>
If X $\sim$ N(2, 1), <b>what is the expected value of -5X?</b> This is denoted as E[-5X].<br>
<b>Answer</b>: -10.<br><br>

<b>9.</b><br>
Let X $\sim$ N(1, 1) and Y $\sim$ N(4, $3^{2}$). <b>What is the value of E[X + Y]?</b><br>
<b>Answer</b>: 5.<br><br>

<b>10.</b><br>
The normal distribution is also linear in the sense that if $X \sim N(\mu, \sigma^{2})$, then for any real constants $a \neq 0$ and b, the distribution of Y = aX + b is distributed N($a \mu + b$, $a^{2} \sigma^{2}$). Using this fact, what is the distribution of $Z = \frac{X - \mu}{\sigma}$?<br>
A. N(1, $\sigma$^{2})<br>
B. N(\mu, \sigma)<br>
C. N(0, 1)<br>
D. N(\mu/\sigma, 1)<br>
E. N(\mu, \sigma^{2})<br>
<b>Answer</b>: C.<br><br>

<b>11.</b><br>
<b>Which of the following random variables would yield the highest value of P(-1 < X < 1)?</b><br>
A. X $\sim$ N(0, 0.1)<br>
B. X $\sim$ N(0, 1)<br>
C. X $\sim$ N(0, 10)<br>
D. X $\sim$ N(0, 100)<br>
<b>Answer</b>: A.<br><br>
</p>

#### 1.3.6 Supplementary material for Lesson 3
<p align="justify">
<b>Expected Values</b><br>
$$E(x) = \sum_{x} x \cdot P(X = x) = \sum_{x} x \cdot f(x)$$
$$E(x) = \int_{-\infty}^{\infty} x \cdot f(x) dx$$
$$E(Z) = E(aX + bY + c) = aE(X) + bE(Y) + c$$

We can also compute expectations of functions of X. For example, suppose g(X) = 2/X. There we have<br>
$$E(g(X)) = \int_{-\infty}^{\infty} g(x)f(x) dx = \int_{-\infty}^{\infty} \frac{2}{x} f(x) dx$$

Note that in general,<br>
$$E(g(X)) \neq g(E(X))$$

Let's say continuous random variable X has a PDF<br>
$$f(x) = 3x^{2} I_{0 \leq x \leq 1}(x)$$

We want to find E(X) and E($X^{2}$)<br>
$$E(X) = \int_{-\infty}^{\infty} x \cdot 3x^{2}I_{0 \leq x \leq 1}(x) dx = \int_{0}^{1} x \cdot 3x^{2} dx = \frac{3}{4}$$
$$E(X^{2}) = \int_{-\infty}^{\infty} x^{2} \cdot 3x^{2}I_{0 \leq x \leq 1}(x) dx = \int_{0}^{1} x^{2} \cdot 3x^{2} dx = \frac{3}{5}$$

<b>Variance</b><br>
$$\text{Var}(X) = \sum_{x} (x - \mu)^{2} \cdot P(X = x)$$
$$\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mu)^{2} \cdot f(x) dx$$

For both discrete and continuous X, a convenient formula for the variance is<br>
$$\text{Var}(X) = E[X^{2}] - (E[X])^{2}$$

Variance has a linear property similar to expectation. Again, let X and Y be random variables<br>
$$\text{Var}(X) = \sigma_{X}^{2}, \quad \text{Var}(Y) = \sigma_{Y}^{2}$$

It is also necessary to assume that X and Y are independent. Suppose we are interested in a new random variable<br>
$$Z = aX + bY + c$$

Thevariance of Z is<br>
$$\text{Var}(Z) = \text{Var}(aX + bY +c) = a^{2}\text{Var}(X) + b^{2}\text{Var}(X) = a^{2}\sigma_{X}^{2} + b^{2} \sigma_{Y}^{2}$$

For example, X has a PDF<br>
$$f(x) = 3x^{2} I_{0 \leq x \leq 1}(x)$$

Its variance<br>
$$\text{Var}(X) = E[X^{2}] - (E[X])^{2} = \frac{3}{5} - (\frac{3}{4})^{2} = \frac{3}{80}$$

<b>Geometric Distributions (Discrete)</b><br>
The geometric distribution is the number of trials needed to get the first success, i.e., the number of Bernoulli events until a success is observed, such as the first head when flipping a coin. It takes values on the positive integers starting with one (since at least one trial is needed to observe a success).<br>
$$X \sim \text{Geo}(p)$$
$$P(X = x \mid p) = p(1 - p)^{x-1}, \quad \text{for x = 1, 2, ...} $$
$$E[X] = \frac{1}{p}$$

For example, <b>what is the probability that we flip a fair coin four times and don’t see any heads?</b>  This is the same as asking what is P(X > 4) where X $\sim$ Geo(0.5).<br>
$$P(X > 4) = 1 - P(X = 1) - P(X = 2) - P(X = 3) = \frac{1}{16}$$

<b>Bernoulli</b><br>
The Bernoulli distribution is used for binary outcomes, coded as 0 and 1. It has one param- eter p, which is the probability of “success” or 1.<br>
$$
\begin{aligned}
& X \sim \text{Bern}(p) \\
& P(X = x \mid p) = p^{x} (1-p)^{1-x}, \quad \text{for } x = 0, 1 \\
& E[X] = p \\
& \text{Var}[X] = p(1-p)
\end{aligned}
$$

One common example is the outcome of flipping a fair coin (p = 0.5).<br><br>

<b>Binomial</b><br>
The binomial distribution counts the number of “successes” in n independent Bernoulli trials (each with the same probability of success). Thus if $X_{1}$, ..., $X_{n}$ are independent Bernoulli(p) random variables, then Y = $\sum_{i=1}^{n} X_{i}$ is binomial distributed.<br>
$$
\begin{aligned}
& Y \sim \text{Binom}(n, p) \\
& P(Y = y \mid n, p) = \binom{n}{y} p^{y} (1-p)^{n-y}, \quad \text{for } y = 0, 1, ..., n \\
& E[X] = np \\
& \text{Var}[Y] = np(1-p)
\end{aligned}
$$

<b>Negative Binomial</b><br>
The negative binomial distribution extends the geometric distribution to model the number of failures before achieving the rth success. It takes values on the positive integers starting with 0.<br>
$$
\begin{aligned}
& Y \sim \text{NegBinom}(r, p) \\
& P(Y = y \mid r, p) = \binom{r + y - 1}{y} p^{r} (1-p)^{y}, \quad \text{for } y = 0, 1, .. \\
& E[Y] = \frac{r(1-p)}{p} \\
& \text{Var}[Y] = \frac{r (1- p)}{p^{2}}
\end{aligned}
$$

Note that the geometric distribution is a special case of the negative binomial distribution where r = 1. Because 0 < p < 1, we have E[Y ] < Var[Y ]. This makes the negative binomial a popular alternative to the Poisson when modeling counts with high variance (recall, that the mean equals the variance for Poisson distributed variables).<br><br>

<b>Multinomial</b><br>
Another generalization of the Bernoulli and the binomial is the multinomial distribution, which is like a binomial when there are more than two possible outcomes. Suppose we have n trials and there are k different possible outcomes which occur with probabilities $p_{1}$, ..., $p_{k}$.<br><br>

For example, we are rolling a six-sided die that might be loaded so that the sides are not equally likely, then n is the total number of rolls, k = 6, $p_{1}$ is the probability of rolling a one, and we denote by $x_{1}$, ..., $x_{6}$ a possible outcome for the number of times we observe rolls of each of one through six, where<br>
$$\sum_{i=1}^{6} x_{i} = n, \quad \sum_{i=1}^{6} p_{i} = 1$$
$$f(x_{1}, ..., x_{k} \mid p_{1}, ..., p_{k}) = \frac{n!}{x_{1}! \cdots x_{k}!} p_{1}^{x_{1}} \cdots p_{k}^{x_{k}}$$

<b>Poisson</b><br>
The Poisson distribution is used for counts, and arises in a variety of situations. The parameter $\lambda$ > 0 is the rate at which we expect to observe the thing we are counting.<br>
$$X \sim \text{Pois}(\lambda)$$
$$P(X = x \mid \lambda) = \frac{\lambda^{x} e^{-\lambda}}{x!}, \quad \text{for i = 0, 1, 2, ...}$$
$$E[X] = \lambda, \quad \text{Var}[X] = \lambda$$

A Poisson process is a process wherein events occur on average at rate λ, events occur one at a time, and events occur independently of each other.<br><br>

Example: Significant earthquakes occur in the Western United States approximately following a Poisson process with rate of two earthquakes per week. <b>What is the probability there will be at least 3 earthquakes in the next two weeks?</b>
The rate per 2 weeks is $2 \times 2$ = 4, so let X $\sim$ Pois(4) and we want to know<br>
$$P(X \geq 3) = 1 - P(X \leq 2) = 1 - P(X = 0) - P(X = 1) - P(X = 2) = 1 - 13e^{-4} = 0.762$$

Note 0! = 1 by definition.<br><br>

<b>Exponential</b><br>
The exponential distribution is often used to model the waiting time between random events. Indeed, if the waiting times between successive events are independent from an Exp(λ) distribution, then for any fixed time window of length t, the number of events occurring in that window will follow a Poisson distribution with mean tλ.<br>
$$X \sim \text{Exp}(\lambda)$$
$$f(x \mid \lambda) = \lambda e^{-\lambda x} I_{x \geq 0}(x)$$
$$E[X] = \frac{1}{\lambda}$$
$$\text{Var}[X] = \frac{1}{\lambda^{2}}$$

Similar to the Poisson distribution, the parameter λ is interpreted as the rate at which the events occur.<br><br>

<b>Double Exponential</b><br>
The double exponential (or Laplace) distribution generalizes the exponential distribution for a random variable that can be positive or negative. The PDF looks like a pair of back-to-back exponential PDFs, with a peak at 0.<br>
$$
\begin{aligned}
& X \sim \text{DExp}(\lambda) \\
& f(x \mid \mu, \tau) = \frac{\tau}{2} e^{-\tau \left | x - \mu \right |} \\
& E[X] = \mu \\
& \text{Var}[X] = \frac{1}{2 \tau^{2}}
\end{aligned}
$$

Most of the probability mass for the double exponential distribution is near 0. For this reason, it is commonly used as a “shrinkage prior” for situations where we have a collection of parameters and believe that many of them should be 0, but we don’t know which ones are 0. This might arise, for example, with the coefficients in multiple regression.<br><br>

<b>Gamma</b><br>
If $X_{1}$, $X_{2}$, ..., $X_{n}$ are independent (and identically distributed Exp(λ)) waiting times between successive events, then the total waiting time for all n events to occur Y = $\sum_{i=1}^{n} X$ will follow a gamma distribution with shape parameter α = n and rate parameter β = λ.<br>
$$Y \sim \text{Gamma}(\alpha, \beta)$$
$$f(y \mid \alpha, \beta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} y^{\alpha -1} e^{-\beta y} I_{y \geq 0}(y)$$
$$E[Y] = \frac{\alpha}{\beta}, \quad \text{Var}[Y] = \frac{\alpha}{\beta^{2}}$$

where $\Gamma(\cdot)$ is the gamma function, a generalization of the factorial function which can accept non-integer arguments. If n is a positive integer, then<br>
$$\Gamma(n) = (n-1)!$$

Note also that α > 0 and β > 0. The exponential distribution is a special case of the gamma distribution with α = 1. The gamma distribution is used to model positive-valued, continuous quantities whose distribution is right-skewed. As α increases, the gamma distribution more closely resembles the normal distribution.<br><br>

<b>Inverse-Gamma</b><br>
The inverse-gamma distribution is the conjugate prior for $\sigma^{2}$ in the normal likelihood with known mean. It is also the marginal prior/posterior for $\sigma^{2}$ in the model of Lesson 10.2. As the name implies, the inverse-gamma distribution is related to the gamma distribution. If X ∼ Gamma(α, β), then the random variable Y = 1/X ∼ Inverse-Gamma(α, β) where<br>
$$
\begin{aligned}
& f(y) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} y^{-(\alpha + 1)} e^{-\frac{\beta}{y}} \mathbf{I}_{y > 0} \\
& E(Y) = \frac{\beta}{\alpha - 1} \text{ for } \alpha > 1
\end{aligned}
$$

The relationship between gamma and inverse-gamma suggest a simple method for simulating draws from the inverse-gamma distribution. First draw X from the Gamma(α,β) distribution and take Y = 1/X, which corresponds to a draw from the Inverse-Gamma(α, β).<br><br>

<b>Uniform</b><br>
The uniform distribution is used for random variables whose possible values are equally likely over an interval. If the interval is (a, b), then the uniform probability density function (PDF) f(x) is flat for all values in that interval and 0 everywhere else.<br>
$$X \sim \text{Uniform}(a, b)$$
$$f(x \mid a, b) = \frac{1}{b-a} I_{a \leq x \leq b} (x)$$
$$E[X] = \frac{a + b}{2}, \quad \text{Var}[X] = \frac{(b - a)^{2}}{12}$$

The standard uniform distribution is obtained when a = 0 and b = 1.<br><br>

<b>Beta</b><br>
The beta distribution is used for random variables which take on values between 0 and 1. it is commonly used to model probabilities.<br>
$$X \sim \text{Beta}(\alpha, \beta)$$
$$f(x \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha-1} (1 - x)^{\beta-1} I_{0 \leq x \leq 1}(x)$$
$$E[X] = \frac{\alpha}{\alpha + \beta}, \quad \text{Var}[X] = \frac{\alpha \beta}{(\alpha + \beta)^{2}(\alpha + \beta + 1)}$$

Note also that α > 0 and β > 0. The standard Uniform(0,1) distribution is a special case of the beta distribution with α = β = 1.<br><br>

<b>Normal</b><br>
The normal, or Gaussian distribution is one of the most important distributions in statistics. It arises as the limiting distribution of sums (and averages) of random variables. This is due to the Central Limit Theorem. Because of this property, the normal distribution is often used to model the “errors,” or unexplained variation of individual observations in regression models. The standard normal distribution is given by<br>
$$Z \sim N(0, 1)$$
$$f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^{2}}{2}}$$
$$E[Z] = 0, \quad \text{Var}[Z] = 1$$

Now consider $X = \sigma Z + \mu$ where $\sigma$ > 0 and $\mu$ is any real constant.
$$E(X) = E(\sigma Z + \mu) = \sigma \cdot 0 + \mu = \mu$$
$$\text{Var}(X) = \text{Var}(\sigma Z + \mu) = \sigma^{2} \text{Var}(Z) + 0 = \sigma^{2}$$

X follows a normal distribution with mean $\mu$ and variance $\sigma^{2}$ (standard deviation $\sigma$) defined as<br>
$$X \sim N(\mu, \sigma^{2})$$
$$f(x \mid \mu, \sigma^{2}) = \frac{1}{\sqrt{2 \pi \sigma^{2}}}e^{-\frac{(x - \mu)^{2}}{2 \sigma^{2}}}$$

The normal distribution is symmetric about the mean μ, and is often described as a “bell- shaped” curve. Although X can take on any real value (positive or negative), more than 99% of the probability mass is concentrated within three standard deviations of the mean.<br><br>

The normal distribution has several desirable properties.<br>
If $X_{1} \sim N(\mu_{1}, \sigma_{1}^{2})$ and $X_{2} \sim N(\mu_{2}, \sigma_{2}^{2})$ are independent, then<br>
$$X_{1} + X_{2} \sim N(\mu_{1} + \mu_{2}, \sigma_{1}^{2} + \sigma_{2}^{2})$$

Consequently, if we take the average of n independent and identically distributed (iid) normal random variables,<br>
$$\bar{X} = \frac{1}{n} \sum_{i = 1}^{n} X_{i}, \quad X_{i} \sim N(\mu, \sigma^{2}) \text{ for i = 1, 2, ..., n}$$
$$\bar{X} \sim N(\mu, \frac{\sigma^{2}}{n})$$

<b>t distribution</b><br>
If we have normal data, we can estimate the mean $\mu$<br>
$$\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \sim N(0, 1)$$

However, we may not know the value of $\sigma$. If we estimate it from data, we can replace it with sample standard deviation<br>
$$S = \sqrt{\sum_{i} \frac{(X_{i} - \bar{X})^{2}}{n - 1}}$$

This causes a standard t distribution with $\nu = n - 1$ degree of freedom.<br>
$$
\begin{aligned}
& Y \sim t_{\nu} \\
& f(y) = \frac{\Gamma(\frac{\nu + 1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\nu \pi}} (1 + \frac{y^{2}}{v})^{-\frac{\nu + 1}{2}} \\
& E[Y] = 0 \quad \text{if } \nu > 1 \\
& \text{Var}[Y] = \frac{\nu}{\nu - 2}, \quad \text{if } \nu > 2
\end{aligned}
$$

The t distribution is symmetric and resembles the normal distribution, but with thicker tails. As the degrees of freedom increase, the t distribution looks more and more like the standard normal distribution.<br><br>

<b>Dirichlet</b><br>
Just as the beta distribution is the conjugate prior for a binomial likelihood, the Dirichlet distribution is the conjugate prior for the multinomial likelihood. It can be thought of as a multivariate beta distribution for a collection of probabilities (that must sum to 1). The probability density function for the random variables $Y_{1}$, ..., $Y_{K}$ with $Y_{k}$ > 0 and $\sum_{k=1}^{K} Y_{k} = 1$ s given by<br>
$$f(y_{1}, y_{2}, ..., y_{K} \mid \alpha_{1}, \alpha_{2}, ..., \alpha_{K}) = \frac{\Gamma(\sum_{k=1}^{K} \alpha_{k})}{\prod_{k=1}^{K} \Gamma(\alpha_{k})} \cdot p_{1}^{\alpha_{1} - 1} \cdot p_{2}^{\alpha_{2} - 1} \cdot ... \cdot p_{K}^{\alpha_{K}-1}$$

where $\alpha_{k}$ > 0 for k = 1,...,K. The expected value for $Y_{i}$ is $\frac{\alpha_{i}}{\sum_{k=1}^{K} \alpha_{k}}$.<br><br>

Example: The example for the multinomial distribution describes an experiment to estimate the probabilities associated with a loaded die. Suppose roll the die many times and model the data as multinomial, where $x_{1}$ is the number of 1’s observed, $x_{2}$ is the number of 2’s observed, etc., and we use a Dirichlet prior for the probabilities $p_{1}$, ..., $p_{6}$ associated with each face of the die. Similar to the binomial-beta model, the posterior distribution for the probabilities is Dirichlet with updated parameters: $\alpha_{1}$ + $x_{1}$, $\alpha_{2}$ + $x_{2}$, . . ., $\alpha_{6}$ + $x_{6}$.<br><br>

<b>Central Limit Theorem</b><br>
The Central Limit Theorem is one of the most important results in statistics, basically saying that with sufficiently large sample sizes, the sample average approximately follows a normal distribution. This underscores the importance of the normal distribution, as well as most of the methods commonly used which make assumptions about the data being normally distributed.<br><br>

In formal mathematical notation, the Central Limit Theorem says: Let $X_{1}$, ..., $X_{n}$ be independent and identically distributed with<br>
$$E(X_{i}) = \mu$$
$$\text{Var}(X_{i}) = \sigma^{2}, \quad 0 < \sigma^{2} < \infty$$

Then
$$\frac{\sqrt{n} (\bar{X} - \mu)}{\sigma} \Rightarrow N(0, 1)$$

That is $\bar{X}_{n}$ is approximately normally distributed with mean $\mu$ and variance $\frac{\sigma^{2}}{n}$.<br><br>

<b>Bayes Theorem for continuous distributions</b><br>
When dealing with a continuous random variable θ, we can write the conditional density for θ given y as:<br>
$$f(\theta \mid y) = \frac{f(y \mid \theta) f(\theta)}{\int f(y \mid \theta) f(\theta) d\theta}$$

Because θ is continuous, we integrate over all possible values of θ in the denominator rather than take the sum over these values.<br><br>
</p>

### 1.4 Frequentist inference
#### 1.4.1 Background for Lesson 4
<p align="justify">
<b>Products and Exponents</b><br>
we can define product notation as
$$\prod_{i=1}^{n} x_{i} = x_{1} \cdot x_{2} \cdot ... \cdot x_{n}$$

Exponents are of the form $a^{x}$ where a (called the base) and x (called the exponent) are any real numbers. Recall that $a^{0}$ = 1. Exponents have the following useful properties
$$a^{x} \cdot a^{y} = a^{x + y}$$
$$(a^{x})^{y} = a^{xy}$$

<b>Natural Logarithm</b>
$$\log(x \cdot y) = \log(x) + \log(y)$$
$$\log(\frac{x}{y}) = \log(x) - \log(y)$$
$$\log(x^{b}) = b \log(x)$$
$$\log(1) = 0$$

<b>Argmax</b><br>
When we want to maximize a function f(x), there are two things we may be interested in:<br>
1. The value f(x) achieves when it is maximized, which we denote $\max_{x} f(x)$.<br>
2. The x-value that results in maximizing f(x), which we denote $x* = \arg\max_{x} f(x)$.<br><br>
</p>

#### 1.4.2 Confidence intervals
<p align="justify">
Under the frequentist paradigm, we view the data as a random sample from some larger, potentially hypothetical population. We can then make probability statements i.e, long-run frequency statements based on this larger population.<br><br>

As an example, we flip a coin 100 times and we get 44 heads and 56 tails. Then, we can view these 100 flips as a random sample from a much larger infinite hypothetical population of flips from this coin.<br><br>

We can say each flip follows a Bernoulli distribution with p (a probability of head). p maybe unknown or fixed because of a particular physical coin.<br>
$$X_{i} \sim B(p)$$

So, we have 2 questions:<br>
<b>What is a best estimate of p?</b><br>
<b>If we get an estimate of p, how much confidence do we have for it?</b><br><br>

CLT (central limit theorem) tells us if we have enough experiments, a sum of any random variable will follow a normal distibution.<br>
$$\sum_{i=1}^{100}X_{i} \sim N(100p, 100p(1-p))$$

In 95% flips, we observe the number of head n is
$$[100p - 1.96 \sqrt{100p(1-p)}, 100p + 1.96 \sqrt{100p(1-p)}]$$

In this case, we onserve 44 heads<br>
$$\sum xi = 44$$

Then, estimate of probability of head
$$\hat{p_{}} = \frac{44}{100} = 0.44$$

Put estimate of p into confidence interval, number of heads [34.3, 53.7]<br><br>

We have 95% confidence that probability of heads is between [0.343, 0.537]<br><br>

In this example of flipping a coin 100 times, observing 44 heads resulted in the following 95% confidence interval for p: (.343, .537). From this we concluded that it is plausible that the coin may be fair because p=.5 is in the interval.<br>
Suppose instead that we flipped the coin 100,000 times, observing 44,000 heads (the same percentage of heads as before). Then using the method just presented, the 95% confidence interval for p is (.437, .443). <b>Is it reasonable to conclude that this is a fair coin with 95% confidence?</b><br>
A. Yes<br>
B. No<br><br>

<b>Answer</b>: B.<br><br>

Here is a question, <b>what does this interval really mean? What does it mean we have 95% confidence?</b><br>
Under frequentist paradigm, what this means is if we repeat this trial an infinite number of times, each time we creat a confidence interval in this way based on the data we observe, on average 95% of the intervals we make will contain the true value of p.<br><br>

Further more, we want to know if a particular interval contains the true value of p. In other word, what is a probability that this interval contains the true value of p? Under the frequentist paradigm, we are assuming there is a fixed right answer for p. So, there is only two situation: p is in the interval or not. From a frequentist perspective, probability of p in the interval is 0 or 1. This is not a particularly satisfying explanation. So, we need Bayesian approach to calculate the probability that p is in the interval.<br><br>

In the coin-flipping example, we could repeat this experiment (100 flips) as many times as we wish. Suppose the coin really is fair ( p= 0.5) and we repeat the experiment a large number of times, each time computing a 95% confidence interval. <b>How many of our intervals, on average, would we expect to contain the true value of 0.5?</b><br>
A. 0%<br>
B. about 5%<br>
C. about 95%<br>
D. 100%<br><br>

<b>Answer</b>: C.<br><br>
</p>

#### 1.4.3 Likelihood function and maximum likelihood
<p align="justify">
Consider a hospital where 400 patients are admitted over a month for heart attack, and a month later 72 of thme have died and 328 have survived. So, we can ask, what is our estimate of mortality rate?<br><br>

Under the frequentist paradigm, we must first establish our reference population. What do we think our reference population is here? One possibility is we could think about heart attack patients in the region. Another is we could think about heart attack patients that are admitted to this hospital, but over a longer period of time. Both of these might be reasonable attempts, <b>but in this case our actual data are not a random sample from either of those populations</b>. We could sort of pretend they are and move on, or we could also try to think harder about what <b>a random sample situation</b> might be. One would be, is we could think about all people in the region who might possibly have a heart attack and might possibly get admitted to this hospital.<br><br>

<b>Q</b>: Suppose we proceed to infer the survival rate of all potential heart attack patients in the region. Because we did not randomly sample from this population, and for other reasons, our inference may be biased or invalid. <b>Which of the following is a potential pitfall to inference in this situation?</b><br>
A. There may be other hospitals in the region whose patients’ demographics are different from those admitted to this particular hospital.<br>
B. Some patients leave the hospital before 30 days for financial reasons and their outcome is unknown.<br>
C. If the original 400 patients were admitted in a relatively short period of time, our inferences may not generalize to other times of the year.<br>
D. All of the above.<br>
E. None of the above.<br><br>

<b>Answer</b>: D.<br><br>

We can say each patient follows Bernoulli distribution with an unknown probability<br>
$$Y_{i} \sim B(\theta)$$
$$P(Y_{i} = 1) = \theta$$

For the entire set of data, probability of all independent Y taking y given a parameter $\theta$<br>
$$P(Y=y \mid \theta) = P(Y_{1} = y_{1}, Y_{2} = y_{2}, ..., Y_{n} = y_{n} \mid \theta) = \prod_{i=1}^{n} P(Y_{i} = y_{i}| \theta)$$
$$L(\theta|y)=\prod_{i=1}^{n}\theta^{y_{i}}(1-\theta)^{1-y_{i}}$$

This is likelihood function. It's about $\theta$ instead of y. Our objective is to maximize this function to acquire a best $\theta$. This is called Maximum likelihhod estimate.<br>
$$\hat{\theta_{}}=arg \max_{\theta} L(\theta|y)$$

We are used to transforming MLE in a format of logarithm<br>
$$
\begin{aligned}
l(\theta) & = logL(\theta|y)=log(\prod_{i=1}^{n}\theta^{y_{i}}(1-\theta)^{1-y_{i}}) \\
& =\sum_{i=1}^{n}(y_{i}log\theta+(1-y_{i})log(1-\theta)) \\
& =log\theta\sum_{i=1}^{n}y_{i}+log(1-\theta)\sum_{i=1}^{n}(1-y_{i})
\end{aligned}
$$

Compute MLE<br>
$$l'(\theta) =\frac{\sum_{i=1}^{n}y_{i}}{\theta}-\frac{\sum_{i=1}^{n}(1-y_{i})}{1-\theta}$$
$$\hat{\theta_{}}=\frac{\sum_{i=1}^{n}y_{i}}{n}=\hat{P_{}} = \frac{72}{400}=0.18$$

Maximum likelihood estimators have many desirable mathematical properties. They're unbiased, they're consistent, and they're invariant.<br><br>

In this case, we can also use CLT to find out an approximate confidence interval<br>
$$\hat{\theta_{}}\pm 1.96\sqrt{\frac{\hat{\theta_{}}(1-\hat{\theta_{}})}{n}}$$

In general, under certain regularity conditions, we can say MLE is approximately normally distributed with a mean at a true value of $\theta$ and a variance of $\frac{1}{I(\hat{\theta_{}})}$<br>
$$\hat{\theta_{}} \sim N(\hat{\theta_{}},\frac{1}{I(\hat{\theta_{}})})$$

The Fisher information is a measure of how much information about $\theta$ is in each data point. It's a function of $\theta$. For Bernoulli random variable, the Fisher information turns out to be<br>
$$I(\theta)=\frac{1}{\theta (1-\theta)}$$

Fisher information is large when $\theta$ is near 0 or 1, while Fisher information is small when $\theta$ is near 0.5. This makes sense, because if you're flipping a coin, and you're getting a mix of heads and tails, that tells you a little bit less than if you're getting nearly all heads or nearly all tails. That's a lot more informative about the value of theta.<br><br>

<b>Q</b>: <b>What is the interpretation of the MLE of θ in the context of the heart attack example?</b><br>
A. The average number of deaths in the 30 day period.<br>
B. The value of the 30-day mortality rate which has the highest likelihood for the data we observed.<br>
C. The life expectancy of the average patient which has the highest likelihood for the data we observed.<br>
D. The maximum number of patients who could survive in the 30 day period.<br><br>

<b>Answer</b>: D.<br><br>
</p>

#### 1.4.4 Computing the MLE
<p align="justify">
Take exponential distribution as an instance
$$
\begin{aligned}
& X_{i}\sim exp(\lambda) \\
& f(x \mid \lambda) = \lambda e^{-\lambda x} \\
& L(\lambda |\mid x) = \prod_{i=1}^{n} \lambda e^{-\lambda x_{i}} = \lambda^{n}e^{-\lambda\sum x_{i}} \\
& l(\lambda) = n \log \lambda - \lambda \sum x_{i} \\
& l'(\lambda) = \frac{n}{\lambda}-\sum x_{i} = 0 \\
& \hat{\lambda_{}} = \frac{n}{\sum x_{i}} = \frac{1}{\bar{X}}
\end{aligned}
$$

This makes sense, because the mean for an exponential distribution is $\frac{1}{\lambda}$, so we take 1 over the sample average.<br><br>

Another example of uniform distribution with an unknown upper bound of $\theta$
$$
\begin{aligned}
& X_{i} \sim U[0, \theta] \\
& f(x \mid \theta) = \frac{1}{\theta}I_{0\leq x_{i}\leq \theta} \\
& L(x \mid \theta) = \prod_{i=1}^{n}\frac{1}{\theta}I_{0\leq x_{i}\leq \theta} = \theta^{-n}I_{0\leq \min_{i}x_{i}\leq\max_{i}x_{i}\leq\theta}
\end{aligned}
$$

In this example, we don't need a log tranform and we take a derivative on $L(\theta)$ with the indicator function hanging around<br>
$$L'(\theta)=-n\theta^{-n-1}I_{0\leq \min_{i}x_{i} \leq \max_{i}x_{i}\leq\theta}$$

For any $\theta>0$, $L(\theta)$ is negative, which means $L(x|\theta)$ always decreases. So when $\theta$ is minimum, $L(x|\theta)$ is maximun.
$$\hat{\theta_{}} = \max_{i}x_{i}$$

<b>Q</b>: <b>What is the distinction between $\theta$ and $\hat{\theta_{}}$?</b><br>
A. $\theta$ is the parameter itself, while $\hat{\theta_{}}$ is the MLE of $\theta$<br>
B. $\hat{\theta_{}}$ is our best guess of the parameter a priori, while $\theta$ is our best guess of the parameter after observing data.<br>
C. $\hat{\theta_{}}$ is our estimate of the likelihood value, while $\theta$ is the maximum value the likelihood can achieve.<br><br>

<b>Answer</b>: A.<br><br>

<b>Q</b>: Suppose we observe n=5 independent draws from a Uniform(0, $\theta$) distribution. They are {0.2,4.6,3.3,4.1,5.2}. <b>What is the MLE for $\theta$?</b><br><br>

<b>Answer</b>: 5.2.<br><br>
</p>

#### 1.4.5 Quiz
<p align="justify">
<b>1. For Questions 1-3, consider the following scenario:</b><br>
In the example of flipping a coin 100 times, suppose that you observe 47 heads and 53 tails. Report the value of $\hat{p_{}}$, the MLE (Maximum Likelihood Estimate) of the probability of obtaining heads.<br><br>

<b>Answer</b>: 0.47.<br><br>

<b>2.</b><br>
Using the central limit theorem as an approximation, and construct a 95% confidence interval for p, the probability of obtaining heads. Report the lower end of this interval and round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.37.<br>
$$\frac{100 \hat{p_{}} \pm 1.96 \sqrt{100 \hat{p_{}} (1 - \hat{p_{}})}}{100} = \hat{p_{}} \pm 1.96 \sqrt{\frac{\hat{p_{}} (1 - \hat{p_{}})}{100}} = 0.47 \pm 1.96 \sqrt{\frac{0.47 \cdot 0.53}{100}}= 0.37$$

<b>3.</b><br>
Report the upper end of this interval and round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.57.<br><br>

<b>4.</b><br>
The likelihood function for parameter θ with data y is based on which of the following?<br>
A. $P(\theta \mid y)$<br>
B. $P(y \mid \theta)$<br>
C. $P(\theta)$<br>
D. $P(y)$<br>
E. None<br><br>

<b>Answer</b>: B.<br><br>

<b>5.</b><br>
If $X_{1}$, $X_{2}$, ..., $X_{n}$ $\sim$ Exponential($\lambda$) (iid means independent and identically distributed), then the MLE for $\lambda$ is $\frac{1}{\bar{X}}$ where $\bar{X}$ is the sample mean. Suppose we observe the following data: $X_{1}$ = 2, $X_{2}$ = 2.5, $X_{3}$ = 4.1, $X_{4}$ = 1.8, $X_{5}$ = 4<br><br>

<b>Answer</b>: 0.35.<br><br>

<b>6.</b><br>
It turns out that the sample mean $\bar{X}$ is involved in the MLE calculation for several models. In fact, if the data are independent and identically distributed from a Bernoulli(p), Poisson(λ), or Normal($\mu$, $\sigma^{2}$), then $\bar{X}$ is the MLE for p, λ, and μ respectively. Suppose we observe n=4 data points from a normal distribution with unknown mean μ. The data are X = {-1.2, 0.5, 0.8, -0.3}. What is the MLE for μ ? Round your answer to two decimal places.<br><br>

<b>Answer</b>: -0.050.<br><br>
</p>

#### 1.4.6 Supplementary material for Lesson 4
<p align="justify">
<b>Derivative of a Likelihood</b><br>
Recall that the likelihood function is viewed as a function of the parameters θ and the data y are considered fixed. Thus, when we take the derivative of the log-likelihood function, it is with respect to θ only.<br><br>

Example: Consider the normal likelihood where only the mean μ is unknown:<br>
$$
\begin{aligned}
f(y \mid \mu) & = \prod_{i=1}^{n} \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{1}{2 \sigma^{2}} (y_{i} - \mu)^{2}} \\
& = \frac{1}{(\sqrt{2 \pi \sigma^{2}})^{n}} e^{-\frac{1}{2 \sigma^{2}} \sum_{i=1}^{n} (y_{i} - \mu)^{2}}
\end{aligned}
$$

which yield the log-likelihood<br>
$$
\begin{aligned}
l(\mu) & = -\frac{n}{2} \log(2 \pi \sigma^{2}) - \frac{1}{2 \sigma^{2}} \sum_{i=1}^{n} (y_{i} - \mu)^{2} \\
& = -\frac{n}{2} \log(2 \pi \sigma^{2}) - \frac{1}{2 \sigma^{2}} \sum_{i=1}^{n} (y_{i}^{2} - 2 \mu y_{i} + \mu^{2}) \\
& = -\frac{n}{2} \log(2 \pi \sigma^{2}) - \frac{1}{2 \sigma^{2}} [\sum_{i=1}^{n} y_{i}^{2} - 2 \mu \sum_{i=1}^{n} y_{i} + n \mu^{2}]
\end{aligned}
$$

We take the derivative of $l(\mu)$ with respect to $\mu$ to obtain<br>
$$\frac{d l(\mu)}{d \mu} = \frac{2 \sum_{i=1}^{n} y_{i}}{2 \sigma^{2}} - \frac{2 n \mu}{2 \sigma^{2}} = \frac{\sum_{i=1}^{n} y_{i}}{\sigma^{2}} - \frac{n \mu}{\sigma^{2}} = 0$$
$$\hat{\mu_{}} = \frac{1}{n} \sum_{i=1}^{n} y_{i} = \bar{y}$$

What if only $sigma$ is unknown?<br>
$$\frac{d l(\sigma^{2})}{d \sigma^{2}} = -\frac{n}{2} 2\pi \frac{1}{2 \sigma^{2}} + \sum_{i=1}^{n} (x_{i} - \mu)^{2} (-\frac{1}{2}) (-1) \frac{1}{(\sigma^{2})^{2}} = -\frac{n}{2 \sigma^{2}} + \sum_{i=1}^{n} (x_{i} - \mu)^{2} \frac{1}{2\sigma^{2}} = 0$$
$$\sigma^{2} = \frac{1}{n} \sum_{i=1}^{n} (x_{i} - \mu)^{2}$$

<b>Products of Indicator Functions</b><br>
Because 0 · 1 = 0, the product of indicator functions can be combined into a single indicator function with a modified condition.<br><br>

Examples: $I_{x < 5} \cdot I_{x \geq 0} = I_{0 \leq x < 5}$<br><br>
</p>

### 1.5 Bayesian inference
#### 1.5.1 Background for Lesson 5
<p align="justify">
<b>Cumulative Distribution Function</b><br>
The cumulative distribution function (CDF) exists for every distribution. We define it for random variable X<br>
$$F(x) = P(X \leq x)$$

For discrete-valued variable X, the CDF is computed with summation<br>
$$F(x) = \sum_{t = -\infty}^{x} f(t), \quad \text{where }f(t) = P(X = t)$$

f(t) is the probability mass function (PMF).<br><br>

If X is continuous, the CDF is computed with an integral with f(t) probability density function (PDF).<br>
$$F(x) = \int_{-\infty}^{x} f(t) dt$$

For example, suppose X $\sim$ Binomial(5, 0.6). Then<br>
$$F(1) = P(X \leq 1) = \sum_{t = -\infty}^{1} f(t) = \sum_{t = -\infty}^{-1} 0 + \sum_{t = 0}^{1} \binom{5}{t} 0.6^{t} (1- 0.6)^{5 - t} \approx 0.087$$

Suppose Y $\sim$ Exp(1)<br>
$$F(2) = P(Y \leq 2) = \int_{t = -\infty}^{2} e^{-t} I_{t \geq 0} dt = \int_{t = 0}^{2} e^{-t} dt \approx 0.865$$

The CDF is convenient for calculating probabilities of intervals. Let a and b be any real numbers with a < b. Then the probability that X falls between a and b is equal to<br>
$$P(a < x \leq b) = p(X \leq b) - P(X \leq a) = F(b) - F(a)$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_5_1_1.png"/></center>
</p>
<p align="justify">
<b>Quantile Function</b><br>
The CDF takes a value for a random variable and returns a probability. Suppose instead that we start with a number between 0 and 1, call it p, and we wish to find the value x so that $P(X \leq x) = p$. The value x which satisfies this equation is called the p quantile (or 100p percentile) of the distribution of X.<br><br>

Example: In a standardized test, the 97th percentile of scores among all test-takers is 23. Then 23 is the score you must achieve on the test in order to score higher than 97% of all test-takers. We could equivalently call q = 23 the .97 quantile of the distribution of test scores.<br><br>

Example: The middle 50% of probability mass for a continuous random variable is found between the .25 and .75 quantiles of its distribution. If Z $\sim$ N(0, 1), then the .25 quantile is −0.674 and the .75 quantile is 0.674. Therefore, P(-0.674 < Z < 0.674) = 0.5.<br><br>

<b>Probability Distributions in R</b><br>
R functions for evaluating the normal distribution
<table class="c">
  <tr><th>Function</th><th>What it does</th></tr>
  <tr><td>dnorm(x, mean, sd)</td><td>Evaluate the PDF at x (mean = $\mu$ and sd = $\sqrt{\sigma^{2}}$)</td></tr>
  <tr><td>pnorm(q, mean, sd)</td><td>Evaluate the CDF at q</td></tr>
  <tr><td>qnorm(p, mean, sd)</td><td>Evaluate the quantile function at p</td></tr>
  <tr><td>rnorm(n, mean, sd)</td><td>Generate n pseudo-random samples from the normal distribution</td></tr>
</table><br><br>
</p>
<p align="justify">
R functions for evaluating the density/mass function for several distributions.<br>
<table class="c">
  <tr><th>Distribution</th><th>Function</th><th>Parameters</th></tr>
  <tr><td>Binomial(n, p)</td><td>dbinom(x, size, prob)</td><td>size = n, prob = p</td></tr>
  <tr><td>Poisson(λ)</td><td>dpois(x, lambda)</td><td>lambda = λ</td></tr>
  <tr><td>Exp(λ)</td><td>dexp(x, rate)</td><td>rate = λ</td></tr>
  <tr><td>Gamma(α, β)</td><td>dgamma(x, shape, rate)</td><td>shape = α, rate = β</td></tr>
  <tr><td>Uniform(a, b)</td><td>dunif(x, min, max)</td><td>min = a, max = b</td></tr>
  <tr><td>Beta(α, β)</td><td>dbeta(x, shape1, shape2)</td><td>shape1 = α, shape2 = β</td></tr>
  <tr><td>N($\mu$, $\sigma^{2}$)</td><td>dnorm(x, mean, sd)</td><td>mean = $\mu$, sd = $\sqrt{\sigma^{2}}$)</td></tr>
  <tr><td>$t_{\nu}$</td><td>dt(x, df)</td><td>df = $\nu$</td></tr>
</table><br><br>
</p>
<p align="justify">
<b>Probability Distributions in Excel</b><br>
Excel functions for evaluating the normal distribution $N(\mu, \sigma^{2})$.<br>
<table class="c">
  <tr><th>Function</th><th>What it does</th></tr>
  <tr><td>NORM.DIST(x, mean, standard_dev, FALSE)</td><td>Evaluate the PDF at x (cumulative = FALSE)</td></tr>
  <tr><td>NORM.DIST(x, mean, standard_dev, TRUE)</td><td>Evaluate the CDF at x (cumulative = TRUE)</td></tr>
  <tr><td>NORM.INV(probability, mean, standard_dev)</td><td>Evaluate the quantile function at probability</td></tr>
  <tr><td>NORM.INV(RAND(), mean, standard_dev)</td><td>Generate one sample (probability=RAND())</td></tr>
</table><br><br>
</p>
<p align="justify">
Excel functions for evaluating the density/mass function for several distributions<br>
<table class="c">
  <tr><th>Distribution</th><th>Function</th><th>Parameters</th></tr>
  <tr><td>Binomial(n, p)</td><td>BINOM.DIST(x, trials, probability_s, FALSE)</td><td>trials = n, probability_s = p</td></tr>
  <tr><td>Poisson(λ)</td><td>POISSON.DIST(x, mean, FALSE)</td><td>mean = λ</td></tr>
  <tr><td>Exp(λ)</td><td>EXPON.DIST(x, lambda, FALSE)</td><td>lambda = λ</td></tr>
  <tr><td>Gamma(α, β)</td><td>GAMMA.DIST(x, alpha, beta, FALSE)</td><td>alpha = a, beta = 1/b</td></tr>
  <tr><td>Beta(α, β)</td><td>BETA.DIST(x, alpha, beta, FALSE)</td><td>alpha = α, beta = β</td></tr>
  <tr><td>N($\mu$, $\sigma^{2}$)</td><td>NORM.DIST(x, mean, standard_dev, FALSE)</td><td>mean = $\mu$, standard_dev = $\sqrt{\sigma^{2}}$)</td></tr>
  <tr><td>$t_{\nu}$</td><td>T.DIST(x, deg_freedom, FALSE)</td><td>deg_freedom = $\nu$</td></tr>
</table><br><br>
</p>

#### 1.5.2 Inference example: frequentist
<p align="justify">
Consider a coin, we have no idea if the coin is fair (p(Head) = 0.5) or loaded (P(Head) = 0.7). But we are allowed to flip it 5 times: 2 heads and 3 tails<br>
$$\theta = \{\text{fair}, \text{loaded}  \}$$
$$X \sim \text{Binomial}(5, ?)$$

Probability of head is<br>
$$
\begin{aligned}
f(x \mid \theta) & =
\begin{cases}
  \binom{5}{x}(0.5)^{5}, \quad \theta = \text{fair}  \\
  \binom{5}{x}(0.7)^{x} (0.3)^{5-x}, \quad \text{otherwise}
\end{cases} \\
& = \binom{5}{x}(0.5)^{5} I_{\theta = \text{fair}} + \binom{5}{x}(0.7)^{x} (0.3)^{5-x} I_{\theta = \text{loaded}}
\end{aligned}
$$

Now we know 2 heads, so x = 2<br>
$$
f(\theta \mid x = 2) =
\begin{cases}
  0.3125, \quad \theta = \text{fair}  \\
  0.1323, \quad \text{otherwise}
\end{cases}
$$

We should regard this coin is fair<br>
$$\text{MLE } \hat{\theta_{}} = \text{fair}$$

<b>Q</b>: When X=2, the MLE in this problem is $\hat{\theta_{}}$ = fair. <b>What is the interpretation of the MLE in this context?</b><br>
A. The MLE is the most likely X we could have observed with five flips, given that the coin is fair.<br>
B. The MLE is the probability that the coin is loaded given that we observed two heads in five flips.<br>
C. The MLE is the most likely X we could have observed with five flips, given that the coin is loaded.<br>
D. The MLE is the θ (coin, either fair or loaded) for which observing two heads in five flips is most likely.<br><br>

<b>Answer</b>: D.<br><br>

Another question is how sure are we?<br>
$$P(\theta = \text{fair} \mid X = 2) = P(\theta = \text{fair}) \in \{0, 1\}$$

This is not a satisfying answer because the probability of fair coin or loaded coin is fixed under frequentist paradigm. We have no idea about the probability of P(\theta = \text{fair}).<br><br>
</p>

#### 1.5.3 Inference example: Bayesian
<p align="justify">
Prior P(loaded) = 0.6<br>
$$f(\theta \mid x) = \frac{f(x \mid \theta) p(\theta)}{\sum_{\theta} f(x \mid \theta) p(\theta)} = \frac{\binom{5}{x} [(0.5)^{5} (0.4) I_{\theta = \text{fair}}+ (0.7)^{x}(0.3)^{5-x}(0.6) I_{\theta = \text{loaded}}]}{\binom{5}{x} [(0.5)^{5} (0.4) + (0.7)^{x} (0.3)^{5-x} (0.6)]}$$

Assign X = 2 for two heads<br>
$$f(\theta \mid X =2) = \frac{0.0125 I_{\theta = \text{fair}} + 0.0079 I_{\theta = \text{loaded}}}{0.0125 + 0.0079} = 0.612 I_{\theta = \text{fair}} + 0.388 I_{\theta = \text{loaded}}$$
$$P(\theta = \text{loaded} \mid X = 2) = 0.388$$

<b>Q</b>: In this example, what is the interpretation of P(θ=loaded)?<br>
A. Your "prior" probability that the coin was loaded, before observing any data.<br>
B. The true probability that the coin was loaded, regardless of data.<br>
C. Your "posterior" probability that the coin was loaded, after observing two heads.<br><br>

<b>Answer</b>: A.<br><br>

<b>Q</b>: In this example, what is the interpretation of P($\theta$ = loaded | X = 2)?<br>
A. Your "prior" probability that the coin was loaded, before observing any data.<br>
B. The true probability that the coin was loaded, regardless of data.<br>
C. Your "posterior" probability that the coin was loaded, after observing two heads.<br><br>

<b>Answer</b>: C.<br><br>

If we have different perspectives, we will get different answers. For example, the prior probability is 0.5. which means we are not sure the coin is loaded or fair. Bot of them have a same possibility. Then we calculate the posterior probability<br>
$$P(\theta = \text{loaded}) = 0.5 \rightarrow P(\theta = \text{loaded} \mid X =2) = 0.297$$

If prior is 0.9<br>
$$P(\theta = \text{loaded}) = 0.9 \rightarrow P(\theta = \text{loaded} \mid X =2) = 0.792$$

This is a subjective approach.<br><br>

<b>Q</b>: Recall that the loaded coin comes up heads 70% of the time on average, and a fair coin comes up heads about 50% of the time. Based on past experience or expert knowledge of your brother's behavior, your prior probability that the coin was loaded was 0.6. After testing the coin five times and observing two heads, the posterior probability that the coin was loaded became 0.388. <b>What effect did these data have on your beliefs about the coin?</b><br>
A. The data favored the hypothesis that the coin was fair, reducing your probability that the coin was loaded.<br>
B. The data favored the hypothesis that the coin was fair, increasing your probability that the coin was loaded.<br>
C. The data favored the hypothesis that the coin was loaded, reducing your probability that the coin was loaded.<br>
D. The data favored the hypothesis that the coin was loaded, increasing your probability that the coin was loaded.<br><br>

<b>Answer</b>: A.<br><br>
</p>

#### 1.5.4 Quiz
<p align="justify">
<b>1. For Questions 1-5, consider the following scenario:</b><br>
You are trying to ascertain your American colleague's political preferences. To do so, you design a questionnaire with five yes/no questions relating to current issues. The questions are all worded so that a "yes" response indicates a conservative viewpoint. Let θ be the unknown political viewpoint of your colleague, which we will assume can only take values θ = conservative or θ=liberal. You have no reason to believe that your colleague leans one way or the other, so you assign the prior P(θ=conservative)=0.5. Assume the five questions are independent and let Y count the number of "yes" responses. If your colleague is conservative, then the probability of a "yes" response on any given question is 0.8. If your colleague is liberal, the probability of a "no" response on any given question is 0.7. What is an appropriate likelihood for this scenario?<br>
$$
\begin{aligned}
& \text{A.} \quad f(y \mid \theta) = \binom{5}{y} (0.8)^{y} (0.2)^{5-y} I_{\theta = \text{conservative}} + \binom{5}{y} (0.3)^{y} (0.7)^{5-y} I_{\theta = \text{libre}} \\
\\
& \text{B.} \quad f(y \mid \theta) = \frac{\theta^{y} e^{-\theta}}{y!} \\
\\
& \text{C.} \quad f(y \mid \theta) = \binom{5}{y} (0.2)^{y} (0.8)^{5-y} \\
\\
& \text{D.} \quad f(y \mid \theta) = \binom{5}{y} (0.3)^{y} (0.7)^{5-y} I_{\theta = \text{conservative}} + \binom{5}{y} (0.8)^{y} (0.2)^{5-y} I_{\theta = \text{libre}} \\
\\
& \text{E.} \quad f(y \mid \theta) = \binom{5}{y} (0.8)^{y} (0.2)^{5-y}
\end{aligned}
$$

<b>Answer</b>: A.<br><br>

<b>2.</b><br>
Suppose you ask your colleague the five questions and he answers "no" to all of them. What is the MLE for $\theta$?<br>
A. $\hat{\theta_{}}$ = conservative<br>
B. $\hat{\theta_{}}$ = liberal<br>
C. None of them. MLE is a number.<br><br>

<b>Answer</b>: B.<br>
$$f(\theta = \text{conservative} \mid y = 0) = \binom{5}{0} (0.8)^{0} (0.2)^{5-0} I_{\theta = \text{conservative}} = 0.2^{5}$$
$$f(\theta = \text{liberal} \mid y = 0) = \binom{5}{0} (0.3)^{0} (0.7)^{5-0} I_{\theta = \text{liberal}} = 0.7^{5}$$

<b>3.</b><br>
Recall that Bayes' theorem gives<br>
$$f(\theta \mid y) = \frac{f(y \mid \theta) f(\theta)}{\sum_{\theta} f(y \mid \theta) f(\theta)}$$

<b>What is the corresponding expression for this problem?</b><br>
$$
\begin{aligned}
& \text{A.} \quad f(\theta \mid y) = \frac{\theta^{y} e^{-\theta} (0.5) / y!}{(0.8)^{y} e^{-8} (0.5)/y! + (0.3)^{y} e^{-3} (0.5)/y!} \\
\\
& \text{B.} \quad f(\theta \mid y) = \frac{\binom{5}{y} (0.8)^{y} (0.2)^{5-y} (0.5)^{2}}{\binom{5}{y} (0.8)^{y} (0.2)^{5-y} (0.5) + \binom{5}{y} (0.3)^{y} (0.7)^{5-y} (0.5)} \\
\\
& \text{C.} \quad f(\theta \mid y) = \frac{\binom{5}{y} (0.8)^{y} (0.2)^{5-y} (0.2) I_{\theta = \text{conservative}} + \binom{5}{y} (0.3)^{y} (0.7)^{5-y} (0.7) I_{\theta = \text{liberal}}}{\binom{5}{y} (0.8)^{y} (0.2)^{5-y} (0.2) + \binom{5}{y} (0.3)^{y} (0.7)^{5-y} (0.7)} \\
\\
& \text{D.} \quad f(\theta \mid y) = \frac{\binom{5}{y} (0.8)^{y} (0.2)^{5-y} (0.5) I_{\theta = \text{conservative}} + \binom{5}{y} (0.3)^{y} (0.7)^{5-y} (0.5) I_{\theta = \text{liberal}}}{\binom{5}{y} (0.8)^{y} (0.2)^{5-y} (0.5) + \binom{5}{y} (0.3)^{y} (0.7)^{5-y} (0.5)} \\
\\
& \text{E.} \quad f(\theta \mid y) = \frac{\binom{5}{y} (0.8)^{y} (0.2)^{5-y} (0.5) + \binom{5}{y} (0.3)^{y} (0.7)^{5-y} (0.5)}{\binom{5}{y} (0.8)^{y} (0.2)^{5-y} (0.5) + \binom{5}{y} (0.3)^{y} (0.7)^{5-y} (0.5)}
\end{aligned}
$$
<b>Answer</b>: D.<br><br>

<b>4.</b><br>
Evaluate the expression in Question 3 for y=0 and report the posterior probability that your colleague is conservative, given that he responded "no" to all of the questions. Round your answer to three decimal places.<br>
<b>Answer</b>: 0.00190.<br>
$$P(\theta = \text{conservative} \mid y = 0) = \frac{(0.2)^{5} (0.5)}{(0.2)^{5} (0.5) + (0.7)^{5} (0.5)}$$

<b>5.</b><br>
Evaluate the expression in Question 3 for y=0 and report the posterior probability that your colleague is liberal, given that he responded "no" to all of the questions. Round your answer to three decimal places.<br>
<b>Answer</b>: 0.998.<br>
$$P(\theta = \text{liberal} \mid y = 0) = \frac{(0.7)^{5} (0.5)}{(0.2)^{5} (0.5) + (0.7)^{5} (0.5)}$$

<b>6. For Questions 6-9, consider again the loaded coin example from the lesson.</b><br>
Recall that your brother has a fair coin which comes up heads 50% of the time and a loaded coin which comes up heads 70% of the time. Suppose now that he has a third coin which comes up tails 70% of the time. Again, you don't know which coin your brother has brought you, so you are going to test it by flipping it 4 times, where X counts the number of heads. Let $\theta$ identify the coin so that there are three possibilities $\theta$ = fair, $\theta$ = loaded favoring heads  $\theta$ = loaded favoring tails. Suppose the prior is now P($\theta$ = fair) = 0.4, P($\theta$ = loaded heads) and P($\theta$ = loaded tails) = 0.3. Our prior probability that the coin is loaded is still 0.6, but we do not know which loaded coin it is, so we split the probability evenly between the two options. <b>What is the form of the likelihood now that we have three options?</b><br>
$$
\begin{aligned}
& \text{A.} \quad f(x \mid \theta) = \binom{4}{x} [0.5^{4} (0.4) I_{\theta = \text{fair}} + (0.7)^{x} (0.3)^{4-x}(0.3) I_{\theta = \text{loaded heads}} + (0.3)^{x} (0.7)^{4-x} (0.3) I_{\theta = \text{loaded tails}}] \\
\\
& \text{B.} \quad f(x \mid \theta) = \binom{4}{x} [0.5^{x} (0.5)^{4-x} I_{\theta = \text{fair}} + (0.3)^{x} (0.7)^{4-x} I_{\theta = \text{loaded heads}} + (0.7)^{x} (0.3)^{4-x} I_{\theta = \text{loaded tails}}] \\
\\
& \text{C.} \quad f(x \mid \theta) = \binom{4}{x} [0.5^{x} (0.5)^{4-x} I_{\theta = \text{fair}} + (0.7)^{x} (0.3)^{4-x} I_{\theta = \text{loaded heads}} + (0.3)^{x} (0.7)^{4-x} I_{\theta = \text{loaded tails}}] \\
\\
& \text{D.} \quad f(x \mid \theta) = \binom{4}{x} [0.5^{4} (0.4) I_{\theta = \text{fair}} + (0.3)^{x} (0.7)^{4-x}(0.3) I_{\theta = \text{loaded heads}} + (0.7)^{x} (0.3)^{4-x} (0.3) I_{\theta = \text{loaded tails}}]
\end{aligned}
$$

<b>Answer</b>: C.<br><br>

<b>7.</b><br>
Suppose you flip the coin four times and it comes up heads twice. What is the MLE for $\theta$?<br>
A. $\hat{\theta_{}}$ = fair<br>
B. $\hat{\theta_{}}$ = loaded heads<br>
C. $\hat{\theta_{}}$ = loaded tails<br>
D. None of them. The MLE is a number<br>
<b>Answer</b>: A.<br>
$$
\begin{aligned}
& f(x = 2 \mid \theta = \text{fair}) = \binom{4}{2} [0.5^{2} (0.5)^{4-2} I_{\theta = \text{fair}} = 0.375 \\
& f(x = 2 \mid \theta = \text{loaded heads}) = \binom{4}{2} [0.7^{2} (0.3)^{4-2} I_{\theta = \text{fair}} = 0.2646 \\
& f(x = 2 \mid \theta = \text{loaded tails}) = \binom{4}{2} [0.3^{2} (0.7)^{4-2} I_{\theta = \text{fair}} = 0.2646
\end{aligned}
$$

<b>8.</b><br>
Suppose you flip the coin four times and it comes up heads twice. What is the posterior probability that this is the fair coin? Round your answer to two decimal places.<br>
<b>Answer</b>: 0.49.<br>
$$f(\theta = \text{fair} \mid x = 2) = \frac{\binom{4}{2} (0.5)^{2} (0.5)^{2} (0.4)}{\binom{4}{2} (0.5)^{2} (0.5)^{2} (0.4) + \binom{4}{2} (0.7)^{2} (0.3)^{2} (0.3) + \binom{4}{2} (0.3)^{2} (0.7)^{2} (0.3)}$$

<b>9.</b><br>
Suppose you flip the coin four times and it comes up heads twice. What is the posterior probability that this is a loaded coin (favoring either heads or tails)? Round your answer to two decimal places. Hint: P($\theta$ = fair | x = 2) = 1 - P($\theta$ = loaded | x = 2), so you can use your answer from the previous question rather than repeat the calculation from Bayes' theorem (both approaches yield the same answer).<br><br>

<b>Answer</b>: 1 - 0.49 = 0.51.<br><br>
</p>

#### 1.5.5 Continuous version of Bayes' theorem
<p align="justify">
$$f(\theta \mid y) = \frac{f(y \mid \theta) f(\theta)}{f(y)} = \frac{f(y \mid \theta) f(\theta)}{\int_{-\infty}^{\infty} f(y \mid \theta) f(\theta) d\theta} = \frac{\text{likelihood} \times \text{prior}}{\text{normalizing constant}}$$

We still take the example of coin. $\theta$ follows a Bernoulli distribution.<br>
$$\theta \sim U[0, 1], \quad f(\theta) = \mathbf{I}_{0 \leq \theta \leq 1}$$

If we have one head<br>
$$f(\theta \mid Y = 1) = \frac{\theta^{1} \cdot (1-\theta)^{0} \cdot \mathbf{I}_{0 \leq \theta \leq 1}}{\int_{-\infty}^{\infty} \theta^{1} \cdot (1-\theta)^{0} \cdot \mathbf{I}_{0 \leq \theta \leq 1} d\theta} = \frac{\theta \cdot \mathbf{I}_{0 \leq \theta \leq 1}}{\int_{0}^{1} \theta d\theta} = 2\theta \cdot \mathbf{I}_{0 \leq \theta \leq 1}$$

<b>Q</b>: <b>Why are we allowed to (temporarily) ignore the normalizing constant when finding a posterior distribution?</b><br>
A. The posterior is a PDF of θ, but θ does not appear in f(y), so the absence of f(y) does not change the form of the posterior.<br>
B. The prior contains the only relevant information about the data y, so that f(y) is no longer necessary.<br>
C. If we kept the normalizing constant, it would cancel with the likelihood because they are both probability density functions of the data y.<br><br>

<b>Answer</b>: A.<br><br>

The versions of Bayes' theorem seen before this lesson were for discrete probabilities with probability mass functions. In this lesson, θ is a continuous quantity which can take on infinitely many values, so the prior (and consequently the posterior) for θ is a probability density function. <b>Which of the following is another distinction between this and previous versions of Bayes' theorem?</b><br>
A. The summation over all values of θ in the denominator is replaced with an integral over all values of θ.<br>
B. There was no normalizing constant in previous versions.<br>
C. The prior is a function of θ rather than a function of y ∣ θ.<br>
D. The likelihood is calculated for infinitely many y values rather than a finite number.<br><br>

<b>Answer</b>: A.<br><br>

<b>Q</b>: Recall that the Bernoulli likelihood takes the form $\theta^{y} (1-\theta)^{1-y}$. Assuming a uniform prior as in the lesson, what is the form of the posterior if we had instead observed Y = 0? That is, find f($\theta$ | Y = 0)<br>
$$
\begin{aligned}
& \text{A.} \quad f(\theta \mid Y = 0) = \frac{\theta^{0} \cdot (1-\theta)^{1} \cdot \mathbf{I}_{0 \leq \theta \leq 1}}{\int_{-\infty}^{\infty} \theta^{0} \cdot (1-\theta)^{1} \cdot \mathbf{I}_{0 \leq \theta \leq 1} d\theta} \\
\\
& \text{B.} \quad f(\theta \mid Y = 0) = \frac{\theta^{1} \cdot (1-\theta)^{0} \cdot \mathbf{I}_{0 \leq \theta \leq 1}}{\int_{-\infty}^{\infty} \theta^{1} \cdot (1-\theta)^{0} \cdot \mathbf{I}_{0 \leq \theta \leq 1} d\theta} \\
\\
& \text{C.} \quad f(\theta \mid Y = 0) = 2\theta \cdot \mathbf{I}_{0 \leq \theta \leq 1} \\
\\
& \text{D.} \quad f(\theta \mid Y = 0) = \frac{1}{2 \theta} \cdot \mathbf{I}_{0 \leq \theta \leq 1}
\end{aligned}
$$

<b>Answer</b>: A.<br><br>
</p>

#### 1.5.6 Posterior intervals
<p align="justify">
<b>Prior probability</b><br>
$$f(\theta) = \mathbf{I}_{0 \leq \theta \leq 1}$$

<b>Posterior probability for observing head</b><br>
$$f(\theta \mid Y = 1) = 2\theta \cdot \mathbf{I}_{0 \leq \theta \leq 1}$$

<b>Prior interval estimates</b><br>
$$P(0.025 < \theta < 0.975) = 0.95$$
$$P(\theta > 0.05) = 0.95$$

<b>Posterior interval estimates</b><br>
$$P(0.025 < \theta < 0.975) = \int_{0.025}^{0.975} 2 \theta = 0.975^{2} - 0.025^{2} = 0.95$$
$$P(\theta > 0.05) = 1 - P(\theta \leq 0.05) = 1 - \int_{0}^{0.05} 2\theta = 1 - 0.05^{2} = 0.9975$$

<b>Equal tailed interval</b><br>
$$P(\theta < q \mid Y = 1) = \int_{0}^{q} 2 \theta d\theta = q^{2}$$
$$P(\sqrt{0.025} < \theta < \sqrt{0.975}) = P(0.158 < \theta < 0.987) = 0.95$$

<b>Highest Posterior Density (HPD)</b><br>
$$P(\theta > \sqrt{0.05} \mid Y = 1) = P(\theta > 0.224 \mid Y = 1) = 0.95$$

The posterior PDF of θ favors high values because we observed data consistent with high values of theta (Y=1). In the previous segment, we found that if we had instead observed Y=0, then the posterior PDF for θ would be<br>
$$f(\theta \mid Y = 0) = 2(1 - \theta) \mathbf{I}_{0 \leq \theta \leq 1}$$

<b>Q</b>: <b>Which of the following is the graph of this PDF?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_5_6_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: B.<br><br>

<b>Q</b>: <b>The shaded region in which of the following graphs corresponds to an equal-tailed interval for the normal distribution?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_5_6_2.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: D.<br>
The probability of being to the right of this interval is equal to the probability of being to the left of the interval.<br><br>

<b>Q</b>: Each of the following graphs depicts a 70% credible interval from a posterior distribution. <b>Which of the intervals represents the highest posterior density (HPD) interval?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_5_6_3.png"/></center>
</p>
<p align="justify">
A. 70% credible interval: $\theta \in$ (0.82, 3.89)<br>
B. 70% credible interval: $\theta \in$ (0.82, 3.89)<br>
C. 70% credible interval: $\theta \in$ (1.08, 7.43)<br><br>

<b>Answer</b>: A.<br>
Of all possible 70% credible intervals, this is the one with the highest posterior density values. It is also the shortest interval which captures 70% of the probability.<br><br>

Frequentist confidence intervals have the interpretation that "If you were to repeat many times the process of collecting data and computing a 95% confidence interval, then on average about 95% of those intervals would contain the true parameter value; however, once you observe data and compute an interval the true value is either in the interval or it is not, but you can't tell which." Bayesian credible intervals have the interpretation that "Your posterior probability that the parameter is in a 95% credible interval is 95%."<br><br>
</p>

#### 1.5.7 Quiz
<p align="justify">
<b>1.</b><br>
We use the continuous version of Bayes’ theorem if:<br>
A. θ is continuous<br>
B. Y is continuous<br>
C. f(y∣θ) is continuous<br>
D. All of the above<br>
E. None of the above<br><br>

<b>Answer</b>: A.<br><br>

<b>2.</b><br>
Consider the coin-flipping example from the lesson. Recall that the likelihood for this experiment was Bernoulli with unknown probability of heads, i.e.,<br>
$$f(y \mid \theta) = \theta^{y} (1 - \theta)^{1-y} \mathbf{I}_{0 \leq \theta \leq 1}$$

and we started with a uniform prior on the interval [0, 1]. After the first flip resulted in heads ($Y_{1}$ = 1), the posterior for θ became<br>
$$f(\theta \mid Y_{1} = 1) = 2\theta \mathbf{I}_{0 \leq \theta \leq 1}$$

Now use this posterior as your prior for θ before the next (second) flip. Which of the following represents the posterior PDF for θ after the second flip also results in heads ($Y_{2}$ = 1)?<br>
$$
\begin{aligned}
& \text{A.} \quad f(\theta \mid Y_{2} = 1) = \frac{\theta (1 - \theta) 2\theta}{\int_{0}^{1} \theta (1 - \theta) 2\theta d\theta} \mathbf{I}_{0 \leq \theta \leq 1} \\
\\
& \text{B.} \quad f(\theta \mid Y_{2} = 1) = \frac{\theta 2\theta}{\int_{0}^{1} \theta 2\theta d\theta} \mathbf{I}_{0 \leq \theta \leq 1} \\
\\
& \text{C.} \quad f(\theta \mid Y_{2} = 1) = \frac{(1 - \theta) 2\theta}{\int_{0}^{1} (1 - \theta) 2\theta d\theta} \mathbf{I}_{0 \leq \theta \leq 1}
\end{aligned}
$$

<b>Answer</b>: B.<br><br>

<b>3.</b><br>
Consider again the coin-flipping example from the lesson. Recall that we used a Uniform(0,1) prior for θ. Which of the following is a correct interpretation of $P(0.3 < \theta < 0.9) = 0.6$?<br>
A. (0.3, 0.9) is a 60% credible interval for $\theta$ before observing any data.<br>
B. (0.3, 0.9) is a 60% credible interval for $\theta$ after observing Y = 1<br>
C. (0.3, 0.9) is a 60% confidence interval fpr $\theta$<br>
D. The posterior probability that $\theta \in$ (0.3, 0.9) is 0.6<br><br>

<b>Answer</b>: A.<br><br>

<b>4.</b><br>
Consider again the coin-flipping example from the lesson. Recall that the posterior PDF for θ, after observing Y = 1, was<br>
$$f(\theta \mid Y = 1) = 2\theta \mathbf{I}_{0 \leq \theta \leq 1}$$

Which of the following is a correct interpretation of<br>
$$P(0.3 < \theta < 0.9 \mid Y = 1) = \int_{0.3}^{0.9} 2\theta d\theta = 0.72$$

A. (0.3, 0.9) is a 72% credible interval for $\theta$ before observing any data<br>
B. (0.3, 0.9) is a 72% credible interval for $\theta$ before after Y = 1<br>
C. (0.3, 0.9) is a 72% confidence interval for $\theta$<br>
D. The prior probability that $\theta \in$ (0.3, 0.9) is 0.72<br><br>

<b>Answer</b>: B.<br><br>

<b>5.</b><br>
Which two quantiles are required to capture the middle 90% of a distribution (thus producing a 90% equal-tailed interval)?<br>
A. .10 and .90<br>
B. .025 and .975<br>
C. .05 and .95<br>
D. 0 and .9<br><br>

<b>Answer</b>: C.<br><br>

<b>6.</b><br>
Suppose you collect measurements to perform inference about a population mean θ. Your posterior distribution after observing data is $\theta \mid y \sim N(0, 1)$. Report the upper end of a 95% equal-tailed interval for θ. Round your answer to two decimal places.<br><br>

<b>Answer</b>: 1.96.<br>
(-1.96, 1.96)<br><br>

<b>7.</b><br>
What does "HPD interval" stand for?<br>
A. Highest partial density interval<br>
B. Highest precision density interval<br>
C. Highest posterior density interval<br>
D. Highest point distance interval<br><br>

<b>Answer</b>: C.<br><br>

<b>8.</b><br>
Each of the following graphs depicts a 50% credible interval from a posterior distribution. Which of the intervals represents the HPD interval?<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_5_7_1.png"/></center>
</p>
<p align="justify">
A. 50% interval $\theta \in$ (0.326, 0.674)<br>
B. 50% interval $\theta \in$ (0.196, 0.567)<br>
C. 50% interval $\theta \in$ (0.500, 1.000)<br>
D. 50% interval $\theta \in$ (0.400, 0.756)<br><br>

<b>Answer</b>: A.<br>
This is the 50% credible interval with the highest posterior density values. It is the shortest possible interval containing 50% of the probability under this posterior distribution.<br><br>
</p>

#### 1.5.8 Supplementary material for Lesson 5
<p align="justify">
<b>Normalizing Constants and Proportionality</b><br>
The full expression for a posterior distribution of some parameter θ is given by<br>
$$f(\theta \mid x) = \frac{f(x \mid \theta) f(\theta)}{\int_{-\infty}^{\infty} f(x \mid \theta) f(\theta) d\theta} \propto f(x \mid \theta) f(\theta)$$

For example, normal distribution $\theta \sim N(m, s^{2})$<br>
$$f(\theta) = \frac{1}{\sqrt{2 \pi s^{2}}} e^{-\frac{1}{2 s^{2}} (\theta - m)^{2}} \propto e^{-\frac{1}{2 s^{2}} (\theta - m)^{2}}$$

To evaluate posterior quantities such as posterior probabilities, we will eventually need to find the normalizing constant. If the integral required is not tractable, we can often still simulate draws from the posterior and approximate posterior quantities. In some cases, we can identify $f(x \mid \theta) f(\theta)$ as being proportional to the PDF of some known distribution.<br><br>
</p>

### 1.6 Priors
#### 1.6.1 Priors and prior predictive distributions
<p align="justify">
$$P(\theta \leq c) \quad \text{for all} c \in R$$

In Bayesian framework, if prior is 0 or 1, posterior is 0 or 1. We should avoid these.<br>
$$P(\theta = \frac{1}{2}) = 1$$
$$f(\theta \mid y) \propto f(y \mid \theta) f(\theta) = f(\theta)$$

<b>Q</b>: Suppose you are tasked with eliciting a prior distribution for θ, the proportion of taxpayers who file their returns after the deadline. After speaking with several tax experts, but before collecting data, you are reasonably confident that θ is greater than 0.05, but less than 0.20. <b>Which of the following prior distributions most accurately reflects theses prior beliefs about θ?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_6_1_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: C.<br><br>

$$f(y) = \int f(y \mid \theta) f(\theta) d\theta = \int f(y, \theta) d\theta$$

f(y, $\theta$) is a joint distribution.<br><br>

Recall the definition of conditional probability<br>
$$P(A, B) = \frac{P(A \cap B)}{P(B)}$$

which in our terms for data analysis is<br>
$$f(y \mid \theta) = \frac{f(y, \theta)}{f(\theta)}$$

If we multiply both sides by f($\theta$), we get<br>
$$f(y, \theta) = f(y \mid \theta) \cdot f(\theta)$$

or that the likelihood times the prior is just the joint distribution for the data and the parameters.<br><br>

Prior predictive intervals are useful because they reveal the consequences of the prior at the data (observation) level. The prior predictive distribution is<br>
$$f(y) = \int f(y, \theta) d\theta$$<br>
</p>

#### 1.6.2 Prior predictive: binomial example
<p align="justify">
Suppose we have 10 data, X is the random variable for counting some number<br>
$$X = \sum_{i=1}^{10} Y_{i}$$

We have a uniform distribution for prior of $\theta$<br>
$$f(\theta) = \mathbf{I}_{0 \leq \theta \leq 1}$$

We predict f(x)<br>
$$f(x) = \int f(x \mid \theta) f(\theta) d\theta = \int_{0}^{1} \frac{10!}{x! (10 - x)!} \theta^{x} (1 - \theta)^{10 - x} (1) d\theta$$

With the help of Beta distribution<br>
$$
\begin{aligned}
& \Gamma(n+1) = n! \\
& Z \sim \text{Beta}(\alpha, \beta) \\
& f(z) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} z^{\alpha - 1} (1 - z)^{\beta - 1}
\end{aligned}
$$

We can easily calculate the integral<br>
$$
\begin{aligned}
f(x) & = \int f(x \mid \theta) f(\theta) d\theta \\
& = \int_{0}^{1} \frac{10!}{x! (10 - x)!} \theta^{x} (1 - \theta)^{10 - x} (1) d\theta \\
& = \int_{0}^{1} \frac{\Gamma(11)}{\gamma(x+1) \Gamma(11-x)} \theta^{(x+1) - 1} (1 - \theta)^{(11 - x) - 1} d\theta \\
& = \frac{\Gamma(11)}{\Gamma(12)} \int_{0}^{1} \frac{\Gamma(12)}{\Gamma(x+1) \Gamma(11 - x)} \theta^{(x+1)-1} (1 - \theta)^{(11-x)-1} d\theta \\
& = \frac{\Gamma(11)}{\Gamma(12)} = \frac{1}{11} \quad \text{for } x \in \{0, 1, ..., 10\}
\end{aligned}
$$

The PDF for the beta distribution is given as:
$$f(z) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha) \Gamma(\beta)} z^{\alpha - 1} (1 - z)^{\beta - 1} \mathbf{I}_{0 \leq x \leq 1}, \quad \alpha > 0, \quad \beta > 0$$

Find the value of the following integral:<br>
$$\int_{0}^{1} \frac{\Gamma(5+3)}{\Gamma(5) \Gamma(3)} z^{5 - 1} (1 - z)^{3 - 1} dz = 1$$

<b>Q</b>: Which of the following plots depicts the probability mass function (PMF) of the prior predictive distribution of X just derived? Recall that we used a binomial likelihood for X and a uniform prior for θ, the probability of success.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_6_2_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: A.<br><br>
</p>

#### 1.6.3 Posterior predictive distribution
<p align="justify">
We want to know the posterior probability given the first observation that a coin is flipped as head. We use posterior of data as prior for second data.<br>
$$f(y_{2} \mid y_{1}) = \int f(y_{2} \mid \theta, y_{1}) f(\theta \mid y_{1}) d\theta$$

Besides, two flips are independent.<br>
$$Y_{2} \perp Y_{1} \rightarrow \int f(y_{2} \mid \theta) f(\theta \mid y_{1}) d\theta$$
$$f(y_{2} \mid Y_{1} = 1) = \int_{0}^{1} \theta^{y_{2}} (1 - \theta)^{1 - y_{2}} 2\theta d\theta = \int_{0}^{1} 2\theta^{y_{2} + 1} (1 - \theta)^{1 - y_{2}} d\theta$$

We calculate $Y_{2}$ = head and $Y_{2}$ = tail respectively<br>
$$P(Y_{2} = 1 \mid Y_{1} = 1) = \int_{0}^{1} 2 \theta^{2} d\theta = \frac{2}{3}$$
$$P(Y_{2} = 0 \mid Y_{1} = 1) = \frac{1}{3}$$

<b>Q</b>: <b>What is the key difference between prior predictive and posterior predictive distributions?</b> Assume that the data $Y_{1}$, ..., $Y_{n}$ are independent.<br>
A. The prior predictive is a function of θ and the posterior predictive is a function of Y.<br>
B. The prior predictive is continuous while the posterior predictive is discrete.<br>
C. The prior predictive is useful in selecting priors and the posterior predictive is useful in selecting posterior distributions.<br>
D. The prior predictive averages (marginalizes) over θ with respect to the prior while the posterior predictive averages with respect to the posterior.<br><br>

<b>Answer</b>: D.<br><br>
</p>

#### 1.6.4 Quiz
<p align="justify">
<b>1. For Questions 1-2, consider the following experiment:</b><br>
Suppose you are trying to calibrate a thermometer by testing the temperature it reads when water begins to boil. Because of natural variation, you take several measurements (experiments) to estimate θ, the mean temperature reading for this thermometer at the boiling point.You know that at sea level, water should boil at 100 degrees Celsius, so you use a precise prior with P(θ=100)=1. You then observe the following five measurements: 94.6 95.4 96.2 94.9 95.9. What will the posterior for θ look like?<br>
A. Most posterior probability will be concentrated near the sample mean of 95.4 degrees Celsius.<br>
B. Most posterior probability will be spread between the sample mean of 95.4 degrees Celsius and the prior mean of 100 degrees Celsius.<br>
C. The posterior will be θ=100 with probability 1, regardless of the data.<br>
D. None of the above.<br><br>

<b>Answer</b>: C.<br><br>

<b>2.</b><br>
Suppose you believe before the experiments that the thermometer is biased high, so that on average it would read 105 degrees Celsius, and you are 95% confident that the average would be between 100 and 110. Which of the following prior PDFs most accurately reflects this prior belief?<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_6_4_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: D.<br><br>

<b>3.</b><br>
Recall that for positive integer n, the gamma function has the following property<br>
$$\Gamma(n) = (n-1)!$$

What is the value of $\Gamma$(6)?<br><br>

<b>Answer</b>: 120.<br><br>

<b>4.</b><br>
Find the value of the normalizing constant, c, which will cause the following integral to evaluate to 1.<br>
$$\int_{0}^{1} c \cdot z^{3} (1 - z) dz$$

Hint: Notice that this is proportional to a beta density. We only need to find the values of the parameters α and β and plug those into the usual normalizing constant for a beta density.<br>
A. $\frac{\Gamma(3 + 1)}{\Gamma(3) \Gamma(1)} = \frac{3!}{2!0!} = 3$<br>
B. $\frac{\Gamma(1)}{\Gamma(z) \Gamma(1-z)} = \frac{0!}{(z-1)!1!}$<br>
C. $\frac{\Gamma(4 + 2)}{\Gamma(4) \Gamma(2)} = \frac{5!}{3!1!} = 20$<br><br>

<b>Answer</b>: C.<br><br>

<b>5.</b><br>
Consider the coin-flipping example from Lesson 5. The likelihood for each coin flip was Bernoulli with probability of heads θ, or<br>
$$f(y \mid \theta) = \theta^{y} (1 - \theta)^{1 - y}, \quad \text{for } y = 0 \text{ or } y = 1$$

and we used a uniform prior on θ. Recall that if we had observed $Y_{1}$ = 0 instead of $Y_{1}$ = 1, the posterior distribution for θ would have been<br>
$$f(\theta \mid Y_{1} = 0) = 2(1 - \theta) \mathbf{I}_{0 \leq \theta \leq 1}$$

Which of the following is the correct expression for the posterior predictive distribution for the next flip $Y_{2} \mid Y_{1}$ = 0?<br>
$$
\begin{aligned}
& \text{A.} \quad f(y_{2} \mid Y_{1} = 0) = \int_{0}^{1} \theta^{y_{2}} (1 - \theta)^{1 - y_{2}} 2(1 - \theta) d\theta, \quad \text{for } y_{2} = 0 \text{ or } y_{2} = 1 \\
\\
& \text{B.} \quad f(y_{2} \mid Y_{1} = 0) = \int_{0}^{1} \theta^{y_{2}} (1 - \theta)^{1 - y_{2}} d\theta, \quad \text{for } y_{2} = 0 \text{ or } y_{2} = 1 \\
\\
& \text{C.} \quad f(y_{2} \mid Y_{1} = 0) = \int_{0}^{1} 2(1 - \theta) d\theta, \quad \text{for } y_{2} = 0 \text{ or } y_{2} = 1 \\
\\
& \text{D.} \quad f(y_{2} \mid Y_{1} = 0) = \int_{0}^{1} 2\theta^{y_{2}} (1 - \theta)^{1 - y_{2}} d\theta, \quad \text{for } y_{2} = 0 \text{ or } y_{2} = 1
\end{aligned}
$$

<b>Answer</b>: A.<br><br>

<b>6.</b><br>
The prior predictive distribution for X when θ is continuous is given by<br>
$$\int f(x \mid \theta) \cdot f(\theta) d\theta$$

The analogous expression when θ is discrete<br>
$$\sum_{\theta} f(x \mid \theta) \cdot f(\theta)$$

Let's return to the example of your brother's loaded coin from Lesson 5. Recall that he has a fair coin where heads comes up on average 50% of the time (p=0.5) and a loaded coin (p=0.7). If we flip the coin five times, the likelihood is binomial:<br>
$$f(x \mid p) = \binom{5}{x} p^{x} (1 - p)^{5 - x}$$

where X counts the number of heads. Suppose you are confident, but not sure that he has brought you the loaded coin, so that your prior is<br>
$$f(p) = 0.9 \mathbf{I}_{p = 0.7} + 0.1 \mathbf{I}_{p = 0.5}$$

Which of the following expressions gives the prior predictive distribution of X?<br>
$$
\begin{aligned}
& \text{A.} \quad f(x) = \binom{5}{x} (0.7)^{x} (0.3)^{5-x} + \binom{5}{x} (0.5)^{x} (0.5)^{5-x} \\
\\
& \text{B.} \quad f(x) = \binom{5}{x} (0.7)^{x} (0.3)^{5-x} (0.5) + \binom{5}{x} (0.5)^{x} (0.5)^{5-x} (0.5) \\
\\
& \text{C.} \quad f(x) = \binom{5}{x} (0.7)^{x} (0.3)^{5-x} (0.1) + \binom{5}{x} (0.5)^{x} (0.5)^{5-x} (0.9) \\
\\
& \text{D.} \quad f(x) = \binom{5}{x} (0.7)^{x} (0.3)^{5-x} (0.9) + \binom{5}{x} (0.5)^{x} (0.5)^{5-x} (0.1)
\end{aligned}
$$

<b>Answer</b>: D.<br><br>
</p>

### 1.7 Bernoulli/binomial data
#### 1.7.1 Bernoulli/binomial likelihood with uniform prior
<p align="justify">
Likelihood function<br>
$$f(y \mid \theta) = \theta^{\sum y_{i}} (1 - \theta)^{n - \sum y_{i}}$$

We take a uniform distribution for prior of $\theta$
$$f(\theta) = \mathbf{I}_{0 \leq \theta \leq 1}$$

Posterior<br>
$$
\begin{aligned}
f(\theta \mid y) & = \frac{f(y \mid \theta) f(\theta)}{\int f(y \mid \theta) f(\theta) d\theta} \\
& = \frac{\theta^{\sum y_{i}} (1 - \theta)^{n - \sum y_{i}} \mathbf{I}_{0 \leq \theta \leq 1}}{\int_{0}^{1} \theta^{\sum y_{i}} (1 - \theta)^{n - \sum y_{i}} \mathbf{I}_{0 \leq \theta \leq 1} d\theta} \\
& = \frac{\theta^{\sum y_{i}} (1 - \theta)^{n - \sum y_{i}} \mathbf{I}_{0 \leq \theta \leq 1}}{\frac{\Gamma(\sum y_{i} + 1) \Gamma(n - \sum y_{i} + 1)}{\Gamma(n + 2)} \int_{0}^{1} \frac{\Gamma(n + 2)}{\Gamma(\sum y_{i} + 1) \Gamma(n - \sum y_{i} + 1)} \theta^{\sum y_{i}} (1 - \theta)^{n - \sum y_{i}} d\theta} \\
& = \frac{\Gamma(n + 2)}{\Gamma(\sum y_{i} + 1) \Gamma(n - \sum y_{i} + 1)} \theta^{\sum y_{i}} (1 - \theta)^{n - \sum y_{i}} \mathbf{I}_{0 \leq \theta \leq 1}
\end{aligned}
$$

$\theta \mid y_{i}$ follows a Beta distribution<br>
$$\theta \mid y_{i} \sim \text{Beta}(\sum y_{i} + 1, n - \sum y_{i} + 1)$$

Return to the example of flipping a coin with unknown probability of heads (θ). If we use a Bernoulli likelihood for each coin flip, i.e.,<br>
$$f(y_{i} \mid \theta) = \theta^{y_{i}} (1 - \theta)^{1 - y_{i}} \mathbf{I}_{0 \leq \theta \leq 1}$$

<b>Q</b>: and a uniform prior for θ, what is the posterior distribution for θ if we observe the following sequence: (H, H, T) where H denotes heads (Y=1) and T denotes tails (Y=0)?<br>
A. Beta(2, 1)<br>
B. Beta(3, 2)<br>
C. Beta(2, 3)<br>
D. Uniform(1, 2)<br>
E. Uniform(2, 3)<br>
F. None<br><br>

<b>Answer</b>: B.<br><br>
</p>

#### 1.7.2 Conjugate priors
<p align="justify">
Uniform distribution is Beta(1, 1) and any Beta distribution is conjugate for Bernoulli distribution.<br><br>

$\theta$ follows a Beta distribution<br>
$$f(\theta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}\mathbf{I}_{0 \leq \theta \leq 1}$$

posterior<br>
$$
\begin{aligned}
f(\theta \mid y_{i}) & \propto f(y \mid \theta) f(\theta) \\
& = \theta^{\sum y_{i}} (1 - \theta)^{n - \sum y_{i}} \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} \theta^{\alpha - 1} (1 - \theta)^{\beta - 1} \mathbf{I}_{0 \leq \theta \leq 1}  \\
& \propto \theta^{\alpha + \sum y_{i} - 1} (1 - \theta)^{\beta + n - \sum y_{i} - 1} \mathbf{I}_{0 \leq \theta \leq 1}
\end{aligned}
$$

$\theta \mid y$ follows also Beta distribution<br>
$$\theta \mid y \sim \text{Beta}(\alpha + \sum y_{i}, \beta + n - \sum y_{i})$$

We repeat the example of flipping a coin with unknown probability of heads (θ) from the previous segment, but with a different prior. If we use a Bernoulli likelihood for each coin flip, i.e.,<br>
$$f(y_{i} \mid \theta) = \theta^{y_{i}} (1 - \theta)^{1 - y_{i}} \mathbf{I}_{0 \leq \theta \leq 1}$$

and a Beta(5, 5) prior for θ, <b>what is the posterior distribution for θ if we observe the following sequence: (H, H, T) where H denotes heads (Y=1) and T denotes tails (Y=0)?</b><br>
A. Beta(7, 6)<br>
B. Beta(2, 1)<br>
C. Beta(1, 2)<br>
D. Beta(5, 5)<br><br>

<b>Answer</b>: A.<br><br>

Suppose that instead of the sequence (H, H, T), we observed (H, T, H). Again, use a Bernoulli likelihood for each coin flip, i.e.,<br>
$$f(y_{i} \mid \theta) = \theta^{y_{i}} (1 - \theta)^{1 - y_{i}} \mathbf{I}_{0 \leq \theta \leq 1}$$

and a Beta(5, 5) prior for θ. <b>What is the posterior distribution for θ now?</b><br>
A. Beta(7, 6)<br>
B. Beta(2, 1)<br>
C. Beta(1, 2)<br>
D. Beta(5, 5)<br><br>

<b>Answer</b>: A.<br><br>

Suppose that instead of observing the sequences (H, H, T) and (H, T, H), we were only told that the experiment resulted in two heads and one tail. In this situation, we can use the binomial likelihood, i.e.<br>
$$f(x \mid \theta) = \binom{3}{x} \theta^{x} (1-\theta)^{3-x}\mathbf{I}_{0 \leq \theta \leq 1}$$

where x = $\sum y_{i}$ Again, use a Beta(5, 5) prior for θ. <b>What is the posterior distribution for θ now?</b><br>
A. Beta(7, 6)<br>
B. Beta(2, 1)<br>
C. Beta(1, 2)<br>
D. Beta(5, 5)<br><br>

<b>Answer</b>: A.<br><br>

<b>Conjugate family</b>: a family of distribution is referred to as conjugate if when we use a member of that family as a prior, we get another member of that family as our posterior. For example, Beta distribution is conjugate to Bernoulli distribution and Binomial distribution.<br><br>

<b>Which of the following describes a modeling situation in which the prior is conjugate?</b><br>
A. Likelihood: Poisson, Prior: gamma, Posterior: gamma<br>
B. Likelihood: exponential, Prior: beta, Posterior: Binomial<br>
C. Likelihood: beta, Prior: Bernoulli, Posterior: beta<br>
D. Likelihood: normal, Prior: gamma, Posterior: normal<br><br>

<b>Answer</b>: A.<br><br>
</p>

#### 1.7.3 Posterior mean and effective sample size
<p align="justify">
Prior follows Beta($\alpha$, $\beta$)<br><br>

Effective sample size of prior is $\alpha + \beta$<br><br>

Mean of Beta is $\frac{\alpha}{\alpha + \beta}$<br><br>

Given such a prior, we have a posterior Beta($\alpha + \sum y_{i}$, $\beta + n - \sum y_{i}$)<br>
Posterior mean<br>
$$\frac{\alpha + \sum y_{i}}{\alpha + \sum y_{i} + \beta + n - \sum y_{i}} = \frac{\alpha + \sum y_{i}}{\alpha + \beta + n} = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \frac{\alpha}{\alpha + \beta}  + \frac{n}{\alpha + \beta + n} \cdot \frac{\sum y_{i}}{n}$$

In other word<br>
$$\text{Posterior mean} = \text{Prior weight} \times \text{Prior mean} + \text{Data weight} \times \text{Data mean}$$

Each of the following beta priors yields a prior mean equal to 0.5. <b>Which has the highest effective prior sample size?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_7_3_1.png"/></center>
</p>
<p align="justify">
A. Beta(2, 2)<br>
B. Beta(10, 10)<br>
C. Beta(75, 75)<br>
D. Beta(500, 500)<br><br>

<b>Answer</b>: D.<br><br>

In frequentist paradigm, 95% CI for $\theta$ is<br>
$$\hat{\theta_{}} \pm 1.96 \sqrt{\frac{\hat{\theta_{}} (1 - \hat{\theta})}{n}}$$

In Bayesian paradigm, we can use posterior to calculate CI (credible interval)<br><br>

Suppose f($\theta$) is a prior, data observed yesterday is $y_{1}$, ..., $y_{n}$, posterior is $\theta \mid y_{1}, ..., y_{n}$. While data onserved today is $y_{n+1}$, ..., $y_{n+m}$. We use yesterday's posterior as today's prior. <b>Note that the calculs is not different between a sequence of data or a batch of data</b><br><br>

Suppose we perform an analysis of binomial data with a conjugate beta prior. Our prior has mean 0.4 and effective sample size 5. We then observe 10 trials with 6 successes. That is, the data mean is 0.6. Let $\theta$* denote the posterior mean for θ, the probability of success. <b>Which is the following is true about θ*?</b><br>
A. $\theta$* $\leq$ 0.4<br>
B. 0.4 < $\theta$* < 0.5<br>
C. $\theta$* = 0.5 exactly<br>
D. 0.5 < $\theta$* < 0.6<br>
E. $\theta$* $\geq$ 0.6<br><br>

<b>Answer</b>: C.<br>
$$\theta* = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \frac{\alpha}{\alpha + \beta} + \frac{n}{\alpha + \beta + n} \cdot \frac{\sum y_{i}}{n} = \frac{8}{15}$$

which is between (0.5, 0.6)<br><br>
</p>

#### 1.7.4 Data analysis example in R
{% highlight R %}
# Suppose we are giving two students a multiple-choice exam with 40 questions, 
# where each question has four choices. We don't know how much the students
# have studied for this exam, but we think that they will do better than just
# guessing randomly. 
# 1) What are the parameters of interest?
# 2) What is our likelihood?
# 3) What prior should we use?
# 4) What is the prior probability P(theta>.25)? P(theta>.5)? P(theta>.8)?
# 5) Suppose the first student gets 33 questions right. What is the posterior
#    distribution for theta1? P(theta1>.25)? P(theta1>.5)? P(theta1>.8)?
#    What is a 95% posterior credible interval for theta1?
# 6) Suppose the second student gets 24 questions right. What is the posterior
#    distribution for theta2? P(theta2>.25)? P(theta2>.5)? P(theta2>.8)?
#    What is a 95% posterior credible interval for theta2?
# 7) What is the posterior probability that theta1>theta2, i.e., that the 
#    first student has a better chance of getting a question right than
#    the second student?

############
# Solutions:

# 1) Parameters of interest are theta1=true probability the first student
#    will answer a question correctly, and theta2=true probability the second
#    student will answer a question correctly.

# 2) Likelihood is Binomial(40, theta), if we assume that each question is 
#    independent and that the probability a student gets each question right 
#    is the same for all questions for that student.

# 3) The conjugate prior is a beta prior. Plot the density with dbeta.
theta=seq(from=0,to=1,by=.01)
plot(theta,dbeta(theta,1,1),type="l")
plot(theta,dbeta(theta,4,2),type="l")
plot(theta,dbeta(theta,8,4),type="l")

# 4) Find probabilities using the pbeta function.
1-pbeta(.25,8,4)
1-pbeta(.5,8,4)
1-pbeta(.8,8,4)

# 5) Posterior is Beta(8+33,4+40-33) = Beta(41,11)
41/(41+11)  # posterior mean
33/40       # MLE

lines(theta,dbeta(theta,41,11))

# plot posterior first to get the right scale on the y-axis
plot(theta,dbeta(theta,41,11),type="l")
lines(theta,dbeta(theta,8,4),lty=2)
# plot likelihood
lines(theta,dbinom(33,size=40,p=theta),lty=3)
# plot scaled likelihood
lines(theta,44*dbinom(33,size=40,p=theta),lty=3)

# posterior probabilities
1-pbeta(.25,41,11)
1-pbeta(.5,41,11)
1-pbeta(.8,41,11)

# equal-tailed 95% credible interval
qbeta(.025,41,11)
qbeta(.975,41,11)

# 6) Posterior is Beta(8+24,4+40-24) = Beta(32,20)
32/(32+20)  # posterior mean
24/40       # MLE

plot(theta,dbeta(theta,32,20),type="l")
lines(theta,dbeta(theta,8,4),lty=2)
lines(theta,44*dbinom(24,size=40,p=theta),lty=3)

1-pbeta(.25,32,20)
1-pbeta(.5,32,20)
1-pbeta(.8,32,20)

qbeta(.025,32,20)
qbeta(.975,32,20)

# 7) Estimate by simulation: draw 1,000 samples from each and see how often 
#    we observe theta1>theta2

theta1=rbeta(1000,41,11)
theta2=rbeta(1000,32,20)
mean(theta1>theta2)


# Note for other distributions:
# dgamma,pgamma,qgamma,rgamma
# dnorm,pnorm,qnorm,rnorm
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 1.7.5 Data analysis example in Excel
{% highlight Excel %}
# Suppose we are giving two students a multiple-choice exam with 40 questions, 
# where each question has four choices. We don't know how much the students
# have studied for this exam, but we think that they will do better than just
# guessing randomly. 
# 1) What are the parameters of interest?
# 2) What is our likelihood?
# 3) What prior should we use?
# 4) What is the prior probability P(theta>.25)? P(theta>.5)? P(theta>.8)?
# 5) Suppose the first student gets 33 questions right. What is the posterior
#    distribution for theta1? P(theta1>.25)? P(theta1>.5)? P(theta1>.8)?
#    What is a 95% posterior credible interval for theta1?
# 6) Suppose the second student gets 24 questions right. What is the posterior
#    distribution for theta2? P(theta2>.25)? P(theta2>.5)? P(theta2>.8)?
#    What is a 95% posterior credible interval for theta2?
# 7) What is the posterior probability that theta1>theta2, i.e., that the 
#    first student has a better chance of getting a question right than
#    the second student?

############
# Solutions:

# 1) Parameters of interest are theta1=true probability the first student
#    will answer a question correctly, and theta2=true probability the second
#    student will answer a question correctly.

# 2) Likelihood is Binomial(40, theta), if we assume that each question is 
#    independent and that the probability a student gets each question right 
#    is the same for all questions for that student.

# 3) The conjugate prior is a beta prior. 
set up columns (starting in Column B): theta  f(theta)  L(theta1)  f(theta1|Y)
start theta at 0.01 in cell B2
> Edit > Fill > Series  -- Columns -- Step .01, Stop 0.99
set prior parameters:  label alpha in A2, value 1 in A3
                       label beta in A4, value 1 in A5
prior density in C3
= (FACT($A$3+$A$5-1)/FACT($A$3-1)/FACT($A$5-1))*B2^($A$3-1)*(1-B2)^($A$5-1)
copy and paste to the rest of Column C
> Insert > Chart > Line
change prior parameters, try alpha=4, beta=2, then try alpha=8, beta=4

# 4) Find probabilities using the BETADIST function.
=1-BETADIST(.25,8,4)
=1-BETADIST(.5,8,4)
=1-BETADIST(.8,8,4)

# 5) Posterior is Beta(8+33,4+40-33) = Beta(41,11)
# posterior mean and MLE
=41/(41+11)
=33/40

L(theta1) in D3
=BINOMDIST(33,40,B2,FALSE)
posterior density in E3
= (FACT(41+11-1)/FACT(41-1)/FACT(11-1))*B2^(41-1)*(1-B2)^(11-1)
> Insert > Chart > Line
plotting together doesn't work well because of difference in scale

# posterior probabilities
=1-BETADIST(.25,41,11)
=1-BETADIST(.5,41,11)
=1-BETADIST(.8,41,11)

# equal-tailed 95% credible interval
=BETAINV(0.025,41,11)
=BETAINV(0.975,41,11)

# 6) Posterior is Beta(8+24,4+40-24) = Beta(32,20)
# posterior mean and MLE
=32/(32+20)
=24/40

L(theta2) in Column F
=BINOMDIST(24,40,B2,FALSE)
f(theta2|Y) in Column G
= (FACT(32+20-1)/FACT(32-1)/FACT(20-1))*B2^(32-1)*(1-B2)^(20-1)
> Insert > Chart > Line

=1-BETADIST(.25,32,20)
=1-BETADIST(.5,32,20)
=1-BETADIST(.8,32,20)

=BETAINV(0.025,32,20)
=BETAINV(0.975,32,20)

# 7) Estimate by simulation: draw 500 samples from each and see how often 
#    we observe theta1>theta2

theta1
=BETAINV(RAND(),41,11)
theta2
=BETAINV(RAND(),32,20)
=IF(H2 > I2, 1, 0)
get sum, divide by 500


# Note for other distributions:
# GAMMA.DIST,GAMMA.INV,GAMMA.INV(RAND(),a,1/b)
# NORM.DIST,NORM.INV,NORM.INV(RAND(),mu,sigma)
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 1.7.6 Quiz
<p align="justify">
<b>1. For Questions 1-5, consider the example of flipping a coin with unknown probability of heads (θ)</b><br>
Suppose we use a Bernoulli likelihood for each coin flip, i.e<br>
$$f(y_{i} \mid \theta) = \theta^{y_{i}} (1 - \theta)^{1 - y_{i}} \mathbf{I}_{0 \leq \theta \leq 1}$$

for $y_{i}$ = 0 or $y_{i}$ = 1, and a uniform prior for θ. What is the posterior distribution for θ if we observe the following sequence: (T, T, T, T) where H denotes heads (Y=1) and T denotes tails (Y=0)?<br>
A. Beta(1, 5)<br>
B. Beta(1, 4)<br>
C. Beta(4, 0)<br>
D. Uniform(0, 4)<br>
E. Beta(0, 4)<br><br>

<b>Answer</b>: A.<br><br>

<b>2.</b><br>
Which of the following graphs depicts the posterior PDF of θ if we observe the sequence (T, T, T, T)? (You may want to use R or Excel to plot the posterior.)<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_7_6_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: A.<br>
theta=seq(from=0,to=1,by=.01)<br>
plot(theta,dbeta(theta,1,5),type="l")<br><br>

<b>3.</b><br>
What is the maximum likelihood estimate (MLE) of θ if we observe the sequence (T, T, T, T)?<br><br>

<b>Answer</b>: 0.<br><br>

<b>4.</b><br>
What is the posterior mean estimate of θ if we observe the sequence (T, T, T, T)? Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.17.<br>
$$\frac{\alpha + \beta}{\alpha + \beta + n} \cdot \frac{\alpha}{\alpha + \beta} + \frac{n}{\alpha + \beta + n} \cdot \frac{\sum y_{i}}{n}$$

<b>5.</b><br>
Use R or Excel to find the posterior probability that θ < 0.5 if we observe the sequence (T,T,T,T). Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.96.<br><br>

<b>6. For Questions 6-9, consider the following scenario:</b><br>
An engineer wants to assess the reliability of a new chemical refinement process by measuring θ, the proportion of samples that fail a battery of tests. These tests are expensive, and the budget only allows 20 tests on randomly selected samples. Assuming each test is independent, she assigns a binomial likelihood where X counts the samples which fail. Historically, new processes pass about half of the time, so she assigns a Beta(2,2) prior for θ (prior mean 0.5 and prior sample size 4). The outcome of the tests is 6 fails and 14 passes. What is the posterior distribution for θ?<br>
A. Beta(14, 6)<br>
B. Beta(16, 8)<br>
C. Beta(6, 20)<br>
D. Beta(6, 14)<br>
E. Beta(8, 16)<br><br>

<b>Answer</b>: E.<br><br>

<b>7.</b><br>
Use R or Excel to calculate the upper end of an equal-tailed 95% credible interval for θ. Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.53.<br>
qbeta(.975,8,16) = 0.5291917<br><br>

<b>8.</b><br>
The engineer tells you that the process is considered promising and can proceed to another phase of testing if we are 90% sure that the failure rate is less than .35. Calculate the posterior probability $P(\theta < 0.35 \mid x)$. In your role as the statistician, would you say that this new chemical should pass?<br>
A. Yes, $P(\theta < 0.35 \mid x) \geq 0.9$<br>
B. No, $P(\theta < 0.35 \mid x) < 0.9$<br><br>

<b>Answer</b>: B.<br>
mean(rbeta(1000,8,16) < 0.35) = 0.573<br><br>

<b>9.</b><br>
It is discovered that the budget will allow five more samples to be tested. These tests are conducted and none of them fail. Calculate the new posterior probability $P(\theta < 0.35 \mid x_{1}, x_{2})$. In your role as the statistician, would you say that this new chemical should pass (with the same requirement as in the previous question)? Hint: You can use the posterior from the previous analysis as the prior for this analysis. Assuming independence of tests, this yields the same posterior as the analysis in which we begin with the Beta(2,2) prior and use all 25 tests as the data.<br>
A. Yes, $P(\theta < 0.35 \mid x_{1}, x_{2}) \geq 0.9$<br>
B. No, $P(\theta < 0.35 \mid x_{1}, x_{2}) < 0.9$<br><br>

<b>Answer</b>: B.<br>
mean(rbeta(1000,8,21) < 0.35) = 0.817<br><br>

<b>10.</b><br>
Let<br>
$$X \mid \theta \sim \text{Binomial}(9, \theta)$$

and assume a Beta($\alpha$, $\beta$) prior for θ. Suppose your prior guess (prior expectation) for θ is 0.4 and you wish to use a prior effective sample size of 5, what values of α and β should you use?<br>
A. $alpha$ = 2, $\beta$ = 3<br>
B. $alpha$ = 4, $\beta$ = 6<br>
C. $alpha$ = 4, $\beta$ = 10<br>
D. $alpha$ = 2, $\beta$ = 5<br><br>

<b>Answer</b>: A.<br><br>
</p>

### 1.8 Poisson data
#### 1.8.1 Poisson data
<p align="justify">
Think about chocolate chip cookies. In mass produced chocolate chip cookies, they make a large amount of dough. They mix in a large number of chips, mix it up really well and then chunk out individual cookies. In this process the number of chips per cookie approximately follows a Poisson distribution.<br><br>

Poisson distribution<br>
$$Y_{i} \sim \text{Pois}(\lambda)$$

Likelihhod function<br>
$$f(y \mid \lambda) = \frac{\lambda^{\sum y_{i}} e^{-n\lambda}}{\prod_{i} y_{i}!}, \quad \text{for } \lambda > 0$$

Gamma prior<br>
$$\lambda \sim \Gamma(\alpha, \beta)$$
$$f(\lambda) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} \lambda^{\alpha - 1} e^{-\beta \lambda}$$

Gamma prior mean is $\frac{\alpha}{\beta}$<br><br>

Posterior<br>
$$f(\lambda \mid y) \propto f(y \mid \lambda) f(\lambda) \propto \lambda^{\sum y_{i}} e^{-n\lambda} \lambda^{\alpha - 1} e^{-\beta \lambda} \propto \lambda^{(\alpha + \sum y_{i}) - 1} e^{-(\beta + n) \lambda}$$

Posterior follows also gamma distribution<br>
$$\Gamma(\alpha + \sum y_{i}, \beta + n)$$

Posterior mean<br>
$$\frac{\alpha + \sum y_{i}}{\beta + n} = \frac{\beta}{\beta + n} \cdot \frac{\alpha}{\beta} + \frac{n}{\beta + n} \cdot \frac{\sum y_{i}}{n}$$

<b>Q</b>: The Poisson likelihood is often used to model count data since Poisson random variables are integer-valued, starting at 0. <b>Which of the following scenarios could we appropriately model with a Poisson likelihood?</b><br>
A. Predicting the number of goals scored in a hockey match<br>
B. Predicting whether your hockey team wins its next match (tie counts as a loss)<br>
C. Predicting the weight of a typical hockey player<br>
D. Predicting the number of wins in a series of three games against a single opponent (ties count as losses)<br><br>

<b>Answer</b>: A.<br><br>

How to choose hyperparameters?<br>
1) prior mean $\frac{\alpha}{\beta}$<br>
-- a) prior std dev $\frac{\sqrt{\alpha}}{\beta}$<br>
-- b) effective sample size $\beta$<br>
2) vague prior<br>
samll $\epsilon$ > 0, $\Gamma(\epsilon, \epsilon)$, posterior mean is<br>
$$\frac{\epsilon + \sum y_{i}}{\epsilon + n} \approx \frac{\sum y_{i}}{n}$$

<b>Q</b>: Each of the following gamma distributions is being considered as a prior for a Poisson mean λ. All have the same mean of 4. <b>Which one expresses the most confidence in this prior mean? Equivalently, which has the greatest effective prior sample size?</b><br>
A. Gamma(1,1/4)<br>
B. Gamma(2,1/2)<br>
C. Gamma(5,5/4)<br>
D. Gamma(20,5)<br><br>

<b>Answer</b>: D.<br><br>
</p>

#### 1.8.2 Quiz
<p align="justify">
<b>1. For Questions 1-8, consider the chocolate chip cookie example from the lesson.</b><br>
As in the lesson, we use a Poisson likelihood to model the number of chips per cookie, and a conjugate gamma prior on λ, the expected number of chips per cookie. Suppose your prior expectation for λ is 8. The conjugate prior with mean 8 and effective sample size of 2 is $\text{Gamma}(a, 2)$. Find the value of a.<br><br>

<b>Answer</b>: 16.<br>
$$\frac{\alpha}{\beta} = \frac{16}{2} = 8$$

<b>2.</b><br>
The conjugate prior with mean 8 and standard deviation 1 is $\text{Gamma}(a, 8)$. Find the value of a.<br><br>

<b>Answer</b>: 64.<br>
$$\frac{\sqrt{\alpha}}{\beta} = 1$$

<b>3.</b><br>
Suppose you are not very confident in your prior guess of 8, so you want to use a prior effective sample size of 1/100 cookies. Then the conjugate prior is $\text{Gamma}(a, 0.01)$. Find the value of a. Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.080.<br><br>

<b>4.</b><br>
Suppose you decide on the prior Gamma(8, 1), which has prior mean 8 and effective sample size of one cookie. We collect data, sampling five cookies and counting the chips in each. We find 9, 12, 10, 15, and 13 chips. What is the posterior distribution for $\lambda$?<br>
A. Gamma(8, 1)<br>
B. Gamma(67, 6)<br>
C. Gamma(6, 67)<br>
D. Gamma(5, 59)<br>
E. Gamma(59, 5)<br>
F. Gamma(1, 8)<br><br>

<b>Answer</b>: B.<br><br>

<b>5.</b><br>
Continuing the previous question, what of the following graphs shows the prior density (dotted line) and posterior density (solid line) of $\lambda$?<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_8_2_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: B.<br><br>

<b>6.</b><br>
Continuing Question 4, what is the posterior mean for $\lambda$? Round your answer to one decimal place.<br><br>

<b>Answer</b>: 11.2.<br><br>

<b>7.</b><br>
Continuing Question 4, use R or Excel to find the lower end of a 90% equal-tailed credible interval for $\lambda$. Round your answer to one decimal place.<br><br>

<b>Answer</b>: 9.0.<br>
qgamma(0.05, 67, 6) = 9.021382<br><br>

<b>8.</b><br>
Continuing Question 4, suppose that in addition to the five cookies reported, we observe an additional ten cookies with 109 total chips. What is the new posterior distribution for $\lambda$, the expected number of chips per cookie? Hint: You can either use the posterior from the previous analysis as the prior here, or you can start with the original Gamma(8,1) prior and update with all fifteen cookies. The result will be the same.<br>
A. Gamma(16, 176)<br>
B. Gamma(11, 109)<br>
C. Gamma(10, 109)<br>
D. Gamma(109, 10)<br>
E. Gamma(176, 16)<br><br>

<b>Answer</b>: E.<br><br>

<b>9. For Questions 9-10, consider the following scenario:</b><br>
A retailer notices that a certain type of customer tends to call their customer service hotline more often than other customers, so they begin keeping track. They decide a Poisson process model is appropriate for counting calls, with calling rate θ calls per customer per day. The model for the total number of calls is then $Y \sim \text{Poisson}(n \cdot t \cdot \theta)$ where n is the number of customers in the group and t is the number of days. That is, if we observe the calls from a group with 24 customers for 5 days, the expected number of calls would be $24 \cdot 5 \cdot \theta = 120 \cdot \theta$. The likelihood for Y is then
$$f(y \mid \theta) = \frac{(n t \theta)^{y} e^{-n t \theta}}{y!} \propto \theta^{y} e^{-n t \theta}$$<br>

This model also has a conjugate gamma prior $\theta \sim \text{Gamma}(a, b)$ which has density (PDF)<br>
$$f(\theta) = \frac{b^{a}}{\Gamma(a)} \theta^{a - 1} e^{-b \theta} \propto \theta^{a-1} e^{-b \theta}$$

Following the same procedure outlined in the lesson, find the posterior distribution for θ.<br>
A. Gamma(a + y, b + nt)<br>
B. Gamma(a + y - 1, b + 1)<br>
C. Gamma(y, nt)<br>
D. Gamma(a + 1, b + y)<br><br>

<b>Answer</b>: A.<br><br>

<b>10.</b><br>
On average, the retailer receives 0.01 calls per customer per day. To give this group the benefit of the doubt, they set the prior mean for θ at 0.01 with standard deviation 0.5. This yields a Gamma($\frac{1}{2500}$, $\frac{1}{25}$) prior for θ. Suppose there are n=24 customers in this particular group of interest, and the retailer monitors calls from these customers for t=5 days. They observe a total of y=6 calls from this group. The following graph shows the resulting Gamma(6.0004,120.04) posterior for θ, the calling rate for this group. The vertical dashed line shows the average calling rate of 0.01.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_8_2_2.png"/></center>
</p>
<p align="justify">
Does this posterior inference for θ suggest that the group has a higher calling rate than the average of 0.01 calls per customer per day?<br>
A. Yes, the posterior mean for θ is twice the average of 0.01.<br>
B. Yes, most of the posterior mass (probability) is concentrated on values of θ greater than 0.01.<br>
C. No, the posterior mean is exactly 0.01.<br>
D. No, most of the posterior mass (probability) is concentrated on values of θ less than 0.01.<br><br>

<b>Answer</b>: B.<br><br>
</p>

### 1.9 Exponential data
#### 1.9.1 Exponential data
<p align="justify">
Waiting time can be modled in exponential distribution<br>
$$
\begin{aligned}
& Y \sim \text{Exp}(\lambda) \\
& f(y \mid \lambda) \sim \lambda e^{-\lambda y} \mathbf{I}_{y \geq 0} \\
& E(y) = \frac{1}{\lambda}, \quad \text{Var}(y) = \frac{1}{\lambda^{2}}
\end{aligned}
$$

Gamma distribution is conjugate to exponential likelihood.<br>
$$\Gamma(100, 1000)$$

So, prior mean is 0.1 and prior standard deviation is 0.01.<br><br>

If the waiting time is 12 minutes<br>
$$
\begin{aligned}
& Y = 12 \\
& f(\lambda \mid y) \propto f(y \mid \lambda) f(\lambda) \sim \lambda e^{-\lambda y} \lambda ^{\alpha - 1} e^{-\beta \lambda} \sim \lambda^{(\alpha + 1) - 1} e^{-(\beta+y) \lambda} \\
& \lambda \mid y \sim \Gamma(\alpha + 1, \beta + y) \\
& \lambda \mid y \sim \Gamma(101, 1012)
\end{aligned}
$$

Posterior mean is<br>
$$\frac{101}{1012} = 0.998$$

We can generalize the result from the lesson to more than one data point. Suppose $Y_{1}$, ..., $Y_{n}$ are independent and identically distributed exponential with mean $\frac{1}{\lambda}$ and assume a Gamma($\alpha$, $\beta$) prior for λ. The likelihood is then<br>
$$f(y \mid \lambda) = \lambda^{n} e^{-\lambda \sum y_{i}}$$

and we can follow the same steps from the lesson to obtain the posterior distribution (try to derive it yourself):
$$\lambda \mid y \sim \text{Gamma} (\alpha + n, \beta + \sum y_{i})$$

<b>Q</b>: <b>What is the prior effective sample size in this model?</b><br>
A. $\beta$<br>
B. $\sum y_{i}$<br>
C. $\alpha$<br>
D. $n$<br>
E. $\alpha + \beta$<br><br>

<b>Answer</b>: C.<br>
The data sample size n is added to α to update the first parameter. Thus α can be interpreted as the sample size equivalent in the prior.<br><br>
</p>

#### 1.9.2 Quiz
<p align="justify">
<b>1. For Questions 1-3, refer to the bus waiting time example from the lesson.</b><br>
Recall that we used the conjugate gamma prior for λ, the arrival rate in busses per minute. Suppose our prior belief about this rate is that it should have mean 1/20 arrivals per minute with standard deviation 1/5. Then the prior is Gamma(a,b) with a=1/16. Find the value of b. Round your answer to two decimal places.<br><br>

<b>Answer</b>: 1.25.<br><br>

<b>2.</b><br>
Suppose that we wish to use a prior with the same mean (1/20), but with effective sample size of one arrival. Then the prior for λ is Gamma(1,20). In addition to the original $Y_{1}$ = 12, we observe the waiting times for four additional busses: $Y_{2}$ = 15, $Y_{3}$ = 8, $Y_{4}$ = 13.5, $Y_{5}$ = 25.Recall that with multiple (independent) observations, the posterior for λ is Gamma(α,β) where α=a+n and $\beta$ = b + $\sum y_{i}$. What is the posterior mean for λ? Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.064.<br><br>

<b>3.</b><br>
Continuing Question 2, use R or Excel to find the posterior probability that λ < 1/10? Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.90.<br>
pgamma(q=1/10, shape=1/16, rate=1.25) = 0.9010846<br><br>

<b>4. For Questions 4-10, consider the following earthquake data:</b><br>
The United States Geological Survey maintains a list of significant earthquakes worldwide. We will model the rate of earthquakes of magnitude 4.0+ in the state of California during 2015. An iid exponential model on the waiting time between significant earthquakes is appropriate if we assume:<br>
1. earthquake events are independent,<br>
2. the rate at which earthquakes occur does not change during the year, and<br>
3. the earthquake hazard rate does not change (i.e., the probability of an earthquake happening tomorrow is constant regardless of whether the previous earthquake was yesterday or 100 days ago).<br><br>

Let $Y_{i}$ denote the waiting time in days between the ith earthquake and the following earthquake. Our model is<br>
$$Y_{i} \sim \text{Exponential}(\lambda)$$

where the expected waiting time between earthquakes is $E(Y) = \frac{1}{\lambda}$ days. Assume the conjugate prior $\lambda \sim \text{Gamma}(a, b)$. Suppose our prior expectation for λ is 1/30, and we wish to use a prior effective sample size of one interval between earthquakes. What is the value of a?<br><br>

<b>Answer</b>: 1.<br><br>

<b>5.</b><br>
What is the value of b?<br>

<b>Answer</b>: 30.<br><br>

<b>6.</b><br>
The significant earthquakes of magnitude 4.0+ in the state of California during 2015 occurred on the following dates (<a href="http://earthquake.usgs.gov/earthquakes/browse/significant.php?year=2015">scources</a>): January 4, January 20, January 28, May 22, July 21, July 25, August 17, September 16, December 30. Recall that we are modeling the waiting times between earthquakes in days. Which of the following is our data vector?<br>
A. y = (3, 16, 8, 114, 60, 4, 23, 30, 105)<br>
B. y = (0, 0, 4, 2, 0, 1, 1, 3)<br>
C. y = (16, 8, 114, 60, 4, 23, 30, 105)<br>
D. y = (3, 16, 8, 114, 60, 4, 23, 30, 105, 1)<br><br>

<b>Answer</b>: C.<br><br>

<b>7.</b><br>
The posterior distribution is<br>
$$\lambda \mid y \sim \text{Gamma}(\alpha, \beta)$$

What is the value of α?<br><br>

<b>Answer</b>: 9.<br><br>

<b>8.</b><br>
The posterior distribution is<br>
$$\lambda \mid y \sim \text{Gamma}(\alpha, \beta)$$

What is the value of β?<br><br>

<b>Answer</b>: 390.<br><br>

<b>9.</b><br>
Use R or Excel to calculate the upper end of the 95% equal-tailed credible interval for λ, the rate of major earthquakes in events per day. Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.040.<br><br>

<b>10.</b><br>
The posterior predictive density for a new waiting time $y^{*}$ in days is:<br>
$$f(y^{*} \mid y) = \int f(y^{*} \mid \lambda) \cdot f(\lambda \mid y) d\lambda = \frac{\beta^{\alpha} \Gamma(\alpha + 1)}{(\beta + y^{*})^{\alpha + 1} \Gamma(\alpha)} \mathbf{I}_{y^{*} \geq 0} = \frac{\beta^{\alpha} \alpha}{(\beta + y^{*}) ^{\alpha+1}} \mathbf{I}_{y^{*} \geq 0}$$

where f(λ ∣ y) is the Gamma(α,β) posterior found earlier. Use R or Excel to evaluate this posterior predictive PDF. Which of the following graphs shows the posterior predictive distribution for $y^{*}$?<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_9_2_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: B.<br><br>
</p>

### 1.10 Normal data
#### 1.10.1 Normal likelihood with variance known
<p align="justify">
$$X_{i} \sim N(\mu, \sigma^{2})$$

Piror for $\mu$<br>
$$\mu \sim N(m, s^{2})$$

Posterior<br>
$$f(\mu \mid x) \propto f(x \mid \mu) f(\mu)$$

$\mu \mid x$ follows a normal distribution<br>
$$\mu \mid x \sim N(\frac{\frac{n \bar{X}}{\sigma^{2}} + \frac{m}{s^{2}}}{\frac{n}{\sigma^{2}} + \frac{1}{s^{2}}}, \frac{1}{\frac{n}{\sigma^{2}}+\frac{1}{s^{2}}})$$

Posterior mean<br>
$$\frac{\frac{n \bar{X}}{\sigma^{2}} + \frac{m}{s^{2}}}{\frac{n}{\sigma^{2}} + \frac{1}{s^{2}}} = \frac{\frac{n}{\sigma^{2}}}{\frac{n}{\sigma^{2}} + \frac{1}{s^{2}}} \bar{X} + \frac{\frac{1}{s^{2}}}{\frac{n}{\sigma^{2}} + \frac{1}{s^{2}}} m = \frac{n}{n +\frac{\sigma^{2}}{s^{2}}} \bar{X} + \frac{\frac{\sigma^{2}}{s^{2}}}{n + \frac{\sigma^{2}}{s^{2}}} m$$

Posterior mean = weighted data mean + weighted prior mean.<br><br>

<b>Q</b>: <b>In which of the following scenarios could we most appropriately use a normal likelihood?</b><br>
A. Predicting your class average score on a standardized test<br>
B. Predicting the number of correct responses on a particular student’s quiz<br>
C. Predicting the percent of correct responses on a particular student’s quiz<br>
D. Predicting whether a student will be given form A or form B of an exam<br>
E. Predicting the number of students in your class who will be in attendance on a given Monday<br><br>

<b>Answer</b>: A.<br><br>

<b>Q</b>: The prior (and posterior) predictive distribution for data is particularly simple in the conjugate normal model. If $Y \mid \theta \sim N(\theta, \sigma^{2})$ and $\theta \sim N(m, s^{2})$, then the marginal distribution for Y, obtained as<br>
$$\int f(y, \theta) d\theta \sim N(m, s^{2} + \sigma^{2})$$

Suppose your data are normally distributed with mean θ and variance 1. You select a normal prior for θ with mean 0 and variance 2. Then the prior predictive distribution for one data point would be N(0,a). Find the value of a.<br><br>

<b>Answer</b>: 3.<br><br>
</p>

#### 1.10.2 Normal likelihood with variance unknown
<p align="justify">
For the case of unknow variance<br>
$$x_{i} \mid \mu, \sigma^{2} \sim N(\mu, \sigma^{2})$$

$\mu$ is dependent of $\sigma$<br>
$$\mu \mid \sigma^{2} \sim N(m, \frac{\sigma^{2}}{w}), \quad \text{effective sample size } w = \frac{\sigma^{2}}{\sigma_{\mu}^{2}}$$

Variance prior<br>
$$\sigma^{2} \sim \Gamma^{-1}(\alpha, \beta)$$

Variance posterior
$$\sigma^{2} \mid x \sim \Gamma^{-1}(\alpha + \frac{n}{2}, \beta + \frac{1}{2} \sum_{i=1}^{n} (x_{i} - \bar{X})^{2} + \frac{n w}{2(n+w)}(\bar{X} - m)^{2})$$

$\mu$ posterior<br>
$$\mu \mid \sigma^{2}, x \sim N(\frac{n \bar{X} + w m}{m + w}, \frac{\sigma^{2}}{n + w})$$
$\mu$ posterior mean<br>
$$\frac{n \bar{X} + wm}{n + w} = \frac{w}{n + w} m + \frac{n}{n + w} \bar{X}$$

If we marginal the $\mu$ posterior over $\sigma^{2}$, we can get the predicted posterior which is a t-distribution<br>
$$\mu \mid x \sim t$$

<b>Q</b>: If you are collecting normal data to make inferences about the mean μ, but you don't know the true variance $\sigma^{2}$ (or standard deviation $\sigma$) of the data, three options available to you are:<br>
1) Fix $\sigma^{2}$ at your best guess.<br>
2)Estimate $\sigma^{2}$ from the data and fix it at this value.<br>
3) Specify a prior for $\sigma^{2}$ and μ to estimate them jointly, as presented in this lesson.<br>
Options 1 and 2 allow you to use the methods of Lesson 10.1 by pretending you know the true value of $\sigma^{2}$. This leads to a simpler posterior calculation for μ. <b>Which of the following is a potential advantage of selecting option 3?</b><br>
A. Option 3 more honestly reflects your uncertainty in $\sigma^{2}$, thereby protecting against overly (and inappropriately) confident inferences.<br>
B. Option 3 makes no assumptions about the distribution of your data, thereby reducing the variance of the estimates and increasing the effective sample size.<br>
C. The prior effective sample size in option 3 is zero.<br>
D. Option 3 will always result in narrower credible intervals for μ.<br><br>

<b>Answer</b>: A.<br><br>
</p>

#### 1.10.3 Quiz
<p align="justify">
<b>1. For Questions 1-6, consider the thermometer calibration problem from the quiz in Lesson 6.</b><br>
Suppose you are trying to calibrate a thermometer by testing the temperature it reads when water begins to boil. Because of natural variation, you take n independent measurements (experiments) to estimate θ, the mean temperature reading for this thermometer at the boiling point. Assume a normal likelihood for these data, with mean θ and known variance $\sigma^{2}$ = 0.25 (which corresponds to a standard deviation of 0.5 degrees Celsius). Suppose your prior for θ is (conveniently) the conjugate normal. You know that at sea level, water should boil at 100 degrees Celsius, so you set the prior mean at $m_{0}$ = 100. If you specify a prior variance $s_{0}^{2}$ for $\theta$, which of the following accurately describes the model for your measurements $Y_{i}$, i = 1, ..., n?<br>
$$
\begin{aligned}
& \text{A.} \quad Y_{i} \mid \sigma^{2} \sim N(100, \sigma^{2}), \quad \sigma^{2} \sim \text{Inverse-Gamma}(0.25, s_{0}^{2}) \\
\\
& \text{B.} \quad Y_{i} \mid \theta, \sigma^{2} \sim N(\theta, \sigma^{2}), \quad \sigma^{2} \sim \text{Inverse-Gamma}(100, s_{0}^{2}) \\
\\
& \text{C.} \quad Y_{i} \mid \theta \sim N(\theta, 100), \quad \theta \sim N(0.25, s_{0}^{2}) \\
\\
& \text{D.} \quad Y_{i} \mid \theta \sim N(100, 0.25), \quad \theta \sim N(\theta, s_{0}^{2}) \\
\\
& \text{E.} \quad Y_{i} \mid \theta \sim N(\theta, 0.25), \quad \theta \sim N(100, s_{0}^{2}) 
\end{aligned}
$$

<b>Answer</b>: E.<br><br>

<b>2.</b><br>
You decide you want the prior to be equivalent (in effective sample size) to one measurement. What value should you select for $s_{0}^{2}$ the prior variance of 
θ? Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.25.<br>
The prior effective sample size is<br>
$$\frac{\sigma^{2}}{s_{0}^{2}} = \frac{0.25}{0.25} = 1$$

<b>3.</b><br>
You collect the following n=5 measurements: (94.6, 95.4, 96.2, 94.9, 95.9). What is the posterior distribution for θ?<br>
A. N(95.41,0.250)<br>
B. N(96.17,0.042)<br>
C. N(95.41,0.042)<br>
D. N(100,0.250)<br>
E. N(96.17,24)<br>
F. N(95.41,24)<br><br>

<b>Answer</b>: B.<br><br>

<b>4.</b><br>
Use R or Excel to find the upper end of a 95% equal-tailed credible interval for θ.<br><br>

<b>Answer</b>: 96.57.<br>
qnorm(p=0.975, mean=96.17, sd=sqrt(0.042)) = 96.57167<br><br>

<b>5.</b><br>
After collecting these data, is it reasonable to conclude that the thermometer is biased toward low values?<br>
A. Yes, we have P($\theta$ < 100 | y) > 0.9999<br>
B. Yes, we have P($\theta$ > 100 | y) > 0.9999<br>
C. No, we have P($\theta$ < 100 | y) < 0.0001<br>
D. No, we have P($\theta$ = 100 | y) = 0<br><br>

<b>Answer</b>: A.<br>
pnorm(q=100, mean=96.17, sd=sqrt(0.042)) = 1<br><br>

<b>6.</b><br>
What is the posterior predictive distribution of a single future observation $Y^{*}$?<br>
A. N(96.17, 0.042)<br>
B. N(100, 0.5)<br>
C. N(95.41, 0.292)<br>
D. N(95.41, 0.5)<br>
E. N(96.17, 0.292)<br><br>

<b>Answer</b>: E.<br><br>

<b>7. For Questions 7-10, consider the following scenario</b><br>
Your friend moves from city A to city B and is delighted to find her favorite restaurant chain at her new location. After several meals, however, she suspects that the restaurant in city B is less generous. She decides to investigate.<br><br>

She orders the main dish on 30 randomly selected days throughout the year and records each meal's weight in grams. You still live in city A, so you assist by performing the same experiment at your restaurant. Assume that the dishes are served on identical plates (measurements subtract the plate's weight), and that your scale and your friend’s scale are consistent. The following histogram shows the 30 measurements from Restaurant B taken by your friend.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_10_3_1.png"/></center>
</p>
<p align="justify">
Is it reasonable to assume that these data are normally distributed?<br>
A. Yes, the distribution appears to follow a bell-shaped curve.<br>
B. Yes, the data are tightly clustered around a single number.<br>
C. No, the first bar to the left of the peak is not equal in height to he first bar to the right of the peak.<br>
D. No, there appear to be a few extreme observations (outliers).<br><br>

<b>Answer</b>: D.<br><br>

<b>8.</b><br>
Your friend investigates the three observations above 700 grams and discovers that she had ordered the incorrect meal on those dates. She removes these observations from the data set and proceeds with the analysis using n = 27.She assumes a normal likelihood for the data with unknown mean μ and unknown variance $\sigma^{2}$. She uses the model presented in Lesson 10.2 where, conditional on $\sigma^{2}$, the prior for μ is normal with mean m and variance $\sigma^{2}$/w. Next, the marginal prior for $\sigma^{2}$ is Inverse-Gamma(a,b). Your friend's prior guess on the mean dish weight is 500 grams, so we set m=500. She is not very confident with this guess, so we set the prior effective sample size w=0.1. Finally, she sets a=3 and b=200. We can learn more about this inverse-gamma prior by simulating draws from it. If a random variable X follows a Gamma(a,b) distribution, then $\frac{1}{X}$ follows an Inverse-Gamma(a,b) distribution. Hence, we can simulate draws from a gamma distribution and take their reciprocals, which will be draws from an inverse-gamma. To simulate 1000 draws in R (replace a and b with their actual values):<br>
</p>
{% highlight R %}
z <- rgamma(n=1000, shape=a, rate=b)
x <- 1/z
{% endhighlight %}
<p align="justify">
To simulate one draw in Excel (replace a and b with their actual values):<br>
</p>
{% highlight Excel %}
= 1 / GAMMA.INV( RAND(), a, 1/b )
{% endhighlight %}
<p align="justify">
where probability=RAND(), alpha=a, and beta=1/b. Then copy this formula to obtain multiple draws. Simulate a large number of draws (at least 300) from the prior for $\sigma^{2}$ and report your approximate prior mean from these draws. It does not need to be exact.<br><br>

<b>Answer</b>: 99.39525.<br>
The actual prior mean for $\sigma^{2}$ is $\frac{b}{a-1}$ = 200/2 = 100. The prior variance for $\sigma^{2}$ is $\frac{b^{2}}{(a-1)^{2}(a-2)}$ = 10000<br><br>

<b>9.</b><br>
With the n = 27 data points, your firend calculates the sample mean $\bar{y}$ = 609.7 and sample variance<br>
$$s^{2} = \frac{1}{n-1} \sum (y_{i} - \bar{y})^{2} = 401.8$$

Using the update formulas from Lesson 10.2, she calculates the following posterior distributions:<br>
$$
\begin{aligned}
& \sigma^{2} \mid y \sim \text{Inverse-Gamma}(a', b') \\
& \mu \mid \sigma^{2}, y \sim N(m', \frac{\sigma^{2}}{w + m})
\end{aligned}
$$

where<br>
$$a' = a + \frac{n}{2} = 3 + \frac{27}{2} = 16.5$$
$$
\begin{aligned}
b' & = b + \frac{n-1}{2} s^{2} + \frac{wn}{2(w+n)} (\bar{y} - m)^{2} \\
& = 200 + \frac{17-1}{2} 401.8 + \frac{0.1 \times 27}{2 (0.1 + 27)} (609.7 - 500)^{2} \\
& = 6022.9
\end{aligned}
$$

w = 0.1, and w + n = 27.1. To simulate draws from this posterior, begin by drawing values for $\sigma^{2}$ from its posterior using the method from the preceding question. Then, plug these values for $\sigma^{2}$ into the posterior for μ and draw from that normal distribution.<br><br>

To simulate 1000 draws in R:<br>
</p>
{% highlight R %}
z <- rgamma(1000, shape=16.5, rate=6022.9)
sig2 <- 1/z
mu <- rnorm(1000, mean=609.3, sd=sqrt(sig2/27.1))
{% endhighlight %}
<p align="justify">
To simulate one draw in Excel:<br>
</p>
{% highlight Excel %}
= 1 / GAMMA.INV( RAND(), 16.5, 1/6022.9 )
{% endhighlight %}
<p align="justify">
gets saved into cell A1 (for example) as the draw for $\sigma^{2}$ Then draw<br>
</p>
{% highlight Excel %}
= 1 / GAMMA.INV( RAND(), 16.5, 1/6022.9 )
{% endhighlight %}
<p align="justify">
where probability=RAND(), mean=609.3, standard_dev=SQRT(A1/27.1), and A1 is the reference to the cell containing the draw for $\sigma^{2}$. Then copy these formulas to obtain multiple draws.<br><br>

We can use these simulated draws to help us approximate inferences for μ and $\sigma^{2}$. For example, we can obtain a 95% equal-tailed credible for μ by calculating the quantiles/percentiles of the simulated values.<br><br>

In R<br>
</p>
{% highlight R %}
quantile(x=mu, probs=c(0.025, 0.975))
{% endhighlight %}
<p align="justify">
In Excel:<br>
</p>
{% highlight Excel %}
= PERCENTILE.INC( A1:A500, 0.025 )
= PERCENTILE.INC( A1:A500, 0.975 )
{% endhighlight %}
<p align="justify">
where array=A1:A500 (or the cells where you have stored samples of μ) and k=0.025 or 0.975.<br><br>

Perform the posterior simulation described above and compute your approximate 95% equal-tailed credible interval for μ. Based on your simulation, which of the following appears to be the actual interval?<br>
A. (602, 617)<br>
B. (582, 637)<br>
C. (608, 610)<br>
D. (245, 619)<br><br>

<b>Answer</b>: A.<br><br>

<b>10.</b><br>
You complete your experiment at Restaurant A with n=30 data points, which appear to be normally distributed. You calculate the sample mean $\bar{y} = 622.8$ and sample variance<br>
$$s^{2} = \frac{1}{n-1} \sum (y_{i} - \bar{y})^{2} = 430.1$$

Repeat the analysis from Question 9 using the same priors and draw samples from the posterior distribution of $\sigma_{A}^{2}$ and $\mu_{A}$ (where the A denotes that these parameters are for Restaurant A).<br><br>

Treating the data from Restaurant A as independent from Restaurant B, we can now attempt to answer your friend's original question: is restaurant A more generous? To do so, we can compute posterior probabilities of hypotheses like $\mu_{A} > \mu_{B}$. This is a simple task if we have simulated draws for $\mu_{A}$ and $\mu_{B}$. For i = 1, ..., N (the number of simulations drawn for each parameter), make the comparison $\mu_{A} > \mu_{B}$ using the ith draw for $\mu_{A}$ and $\mu_{B}$. Then count how many of these return a TRUE value and divide by N, the total number of simulations.<br><br>

In R (using 1000 simulated values):<br>
</p>
{% highlight R %}
sum( muA > muB ) / 1000
{% endhighlight %}
<p align="justify">
Or<br>
</p>
{% highlight R %}
mean( muA > muB )
{% endhighlight %}
<p align="justify">
In Excel (for one value):<br>
</p>
{% highlight Excel %}
= IF(A1 > B1, 1, 0)
{% endhighlight %}
<p align="justify">
where the first argument is the logical test which compares the value of cell A1 with that of B1, 1=value_if_true, and 0=value_if_false. Copy this formula to compare all $\mu_{A}$, $\mu_{B}$ pairs. This will yield a column of binary (0 or 1) values, which you can sum or average to approximate the posterior probability.<br><br>

Would you conclude that the main dish from restaurant A weighs more than the main dish from restaurant B on average?<br>
A. Yes, the posterior probability that $\mu_{A}$ > $\mu_{B}$ is at least 0.95<br>
B. Yes, the posterior probability that $\mu_{A}$ > $\mu_{B}$ is les than 0.05<br>
C. No, the posterior probability that $\mu_{A}$ > $\mu_{B}$ is at least 0.95<br>
D. No, the posterior probability that $\mu_{A}$ > $\mu_{B}$ is les than 0.95<br><br>

<b>Answer</b>: A.<br><br>
</p>

#### 1.10.4 Supplementary material for Lesson 10
<p align="justify">
<b>Conjugate Posterior for the Normal Mean</b><br>
Here we derive the update formula from Lesson 10.1 where the likelihood is normal and we use a conjugate normal prior on the mean. Specifically,the model is<br>
$$
\begin{aligned}
& x_{1}, ..., x_{n} \sim N(\mu, \sigma_{0}^{2}) \\
& \mu \sim N(m_{0}, s_{0}^{2})
\end{aligned}
$$

with $\sigma_{0}^{2}$, $m_{0}$ and $s_{0}^{2}$ known. First consider the case in which we have only one data point x. The posterior is then<br>
$$
\begin{aligned}
f(\mu \mid x) & = \frac{f(x \mid \mu) f(\mu)}{\int_{-\infty}^{\infty} f(x \mid \mu) f(\mu) d\mu} \propto f(x \mid \mu) f(\mu) \\
& = \frac{1}{\sqrt{2 \pi \sigma_{0}^{2}}} e^{-\frac{1}{2 \sigma_{0}^{2}} (x - \mu)^{2}} \frac{1}{\sqrt{2 \pi s_{0}^{2}}} e^{-\frac{1}{2 s_{0}^{2}} (\mu - m_{0})^{2}} \\
& \propto e^{-\frac{1}{2 \sigma_{0}^{2}} (x - \mu)^{2}} e^{-\frac{1}{2 s_{0}^{2}} (\mu - m_{0})^{2}} \\
& = e^{-\frac{1}{2 \sigma_{0}^{2}} (x - \mu)^{2} - \frac{1}{2 s_{0}^{2}}(\mu - m_{0})^{2}} \\
& = e^{-\frac{1}{2}[\frac{\mu^{2}}{\sigma_{0}^{2}} + \frac{\mu^{2}}{s_{0}^{2}} + \frac{-2x}{\sigma_{0}^{2}} \mu + \frac{-2 m_{0}}{s_{0}^{2}} \mu + \frac{x^{2}}{\sigma_{0}^{2}} + \frac{m_{0}^{2}}{s_{0}^{2}}]} \\
& = e^{-\frac{1}{2}[(\frac{1}{\sigma_{0}^{2}} + \frac{1}{s_{0}^{2}})\mu^{2} - 2 (\frac{x}{\sigma_{0}^{2}} + \frac{m_{0}}{s_{0}^{2}})\mu + \frac{x^{2}}{\sigma_{0}^{2}} + \frac{m_{0}^{2}}{s_{0}^{2}}]} \\
& = e^{-\frac{1}{2}[(\frac{1}{\sigma_{0}^{2}} + \frac{1}{s_{0}^{2}})\mu^{2} - 2 (\frac{x}{\sigma_{0}^{2}} + \frac{m_{0}}{s_{0}^{2}})\mu]} \cdot e^{-\frac{1}{2} [\frac{x^{2}}{\sigma_{0}^{2}} + \frac{m_{0}^{2}}{s_{0}^{2}}]} \\
& \propto e^{-\frac{1}{2}[(\frac{1}{\sigma_{0}^{2}} + \frac{1}{s_{0}^{2}})\mu^{2} - 2 (\frac{x}{\sigma_{0}^{2}} + \frac{m_{0}}{s_{0}^{2}})\mu]} \\
& = e^{-\frac{1}{2} [\frac{1}{s_{1}^{2}} \mu^{2} - 2(\frac{x}{\sigma_{0}^{2}} + \frac{m_{0}}{s_{0}^{2}}) \mu]} , \quad \text{where } s_{1}^{2} = \frac{1}{\frac{1}{s_{0}^{2}} + \frac{1}{\sigma_{0}^{2}}} \\
& = e^{-\frac{1}{2} [\frac{1}{s_{1}^{2}} \mu^{2} - 2 \frac{s_{1}^{2}}{s_{1}^{2}} (\frac{x}{\sigma_{0}^{2}} + \frac{m_{0}}{s_{0}^{2}}) \mu]} \\
& = e^{-\frac{1}{2 s_{1}^{2}} [\mu^{2} - 2 m_{1} \mu]}, \quad m_{1} = s_{1}^{2} (\frac{m_{0}}{s_{0}^{2}} + \frac{x}{\sigma_{0}^{2}})
\end{aligned}
$$

The next step is to “complete the square” in the exponent:<br>
$$
\begin{aligned}
f(\mu \mid x) & \propto e^{-\frac{1}{2 s_{1}^{2}} [\mu^{2} - 2 m_{1} \mu]} \\
& = e^{-\frac{1}{2 s_{1}^{2}} [\mu^{2} - 2 m_{1} \mu + m_{1}^{2} - m_{1}^{2}]} \\
& = e^{-\frac{1}{2 s_{1}^{2}} [\mu^{2} - 2 m_{1} \mu + m_{1}^{2}]} \cdot e^{\frac{m_{1}^{2}}{2 s_{1}^{2}}} \\
& \propto e^{-\frac{1}{2 s_{1}^{2}} [\mu^{2} - 2 m_{1} \mu + m_{1}^{2}]} \\
& = e^{-\frac{1}{ 2 s_{1}^{2}} (\mu - m_{1})^{2}}
\end{aligned}
$$

which, except for a normalizing constant not involving μ, is the PDF of a normal distribution with mean $m_{1}$ and variance $s_{1}^{2}$.<br><br>

The final step is to extend this result to accommodate n independent data points. The likelihood in this case is<br>
$$
\begin{aligned}
f(x \mid \mu) & = \prod_{i=1}^{n} \frac{1}{\sqrt{2 \pi \sigma_{0}^{2}}} e^{-\frac{1}{2 \sigma_{0}^{2}} (x_{i} - \mu)^{2}} \\
& = (2 \pi \sigma_{0}^{2})^{-\frac{n}{2}} e^{-\frac{1}{2 \sigma_{0}^{2}} \sum_{i=1}^{n} (x_{i} - \mu)^{2}} \\
& \propto e^{-\frac{1}{2 \sigma_{0}^{2}} [\sum_{i=1}^{n} x_{i}^{2} - 2\mu \sum_{i=1}^{n} x_{i} + n \mu^{2}]} \\
& \propto e^{-\frac{1}{2 \sigma_{0}^{2}} [-2n\bar{x}\mu + n \mu^{2}]}
\end{aligned}
$$

We can repeat the steps above or notice that the data contribute only through the sample mean $\bar{x}$ (and n which we assume is known). This means that $\bar{x}$ is a “sufficient statistic” for μ, allowing us to use the distribution of $\bar{x}$ as the likelihood (analogous to using a binomial likelihood in place of a sequence of Bernoullis). The model then becomes<br>
$$\bar{x} \mid \mu \sim N(\mu, \frac{\sigma_{0}^{2}}{n}), \quad \mu \sim N(m_{0}, s_{0}^{2})$$
We now apply our result derived above, replacing x with $\bar{x}$ and $\sigma_{0}^{2}$ with $\frac{\sigma_{0}^{2}}{n}$. This yields the update equation presented in Lesson 10.1.<br><br>

<b>Marginal Distribution of Normal Mean in Conjugate Model</b><br>
Consider again the model x | $\mu$ $\sim$ N($\mu$, $\sigma_{0}^{2}$), $\mu$ $\sim$ N($m_{0}$, $s_{0}^{2}$) with $\sigma_{0}^{2}$ known. Here we derive the marginal distribution for data given by<br>
$$\int_{-\infty}^{\infty} f(x \mid \mu) f(\mu) d\mu$$

This is the prior predictive distribution for a new data point $x^{*}$.<br><br>

To do so, re-write the model in an equivalent, but more convenient form: x = $\mu$ + $\epsilon$ where $\epsilon \sim$ N(0, $\sigma_{0}^{2}$) and $\mu$ = $m_{0}$ + $\eta$ where $\epsilon$ and $eta$ are independent. Now substitute μ into the first equation to get x = $m_{0}$ + $\eta$ + $\epsilon$. Recall that adding two normal random variables results in another normal random variable, so x is normal with<br>
$$
\begin{aligned}
& E(x) = E(m_{0} + \eta + \epsilon) = m_{0} + E(\eta) + E(\epsilon) = m_{0} + 0 + 0 \\
& \text{Var}(x) = \text{Var}(m_{0} + \eta + \epsilon) = \text{Var}(m_{0}) + \text{Var}(\eta) + \text{Var}(\epsilon)  = 0 + s_{0}^{2} + \sigma_{0}^{2}
\end{aligned}
$$

note that we can add variances because of the independence of $\eta$ and $\epsilon$. Therefore the marginal distribution for x is normal with mean $m_{0}$ and variance $s_{0}^{2} + \sigma_{0}^{2}$. The posterior predictive distribution is the same, but with $m_{0}$ and $\sigma_{0}^{2}$ replaced by the posterior updates given in Lesson 10.1.<br><br>

<b>Inverse-Gamma Distribution</b><br>
The inverse-gamma distribution is the conjugate prior for $\sigma^{2}$ in the normal likelihood with known mean. It is also the marginal prior/posterior for $\sigma^{2}$ in the model of Lesson 10.2. As the name implies, the inverse-gamma distribution is related to the gamma distribution. If X $\sim$ Gamma($\alpha$, $\beta$), then the random variable Y = $\frac{1}{X}$ $\sim$ Inverse-Gamma($\alpha$, $\beta$) where<br>
$$f(y) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} y^{-(\alpha+1)} e^{-\frac{\beta}{y}} \mathbf{I}_{y \geq 0}$$
$$E(Y) = \frac{\beta}{\alpha - 1}, \quad \text{for } \alpha > 1$$

The relationship between gamma and inverse-gamma suggest a simple method for simulating draws from the inverse-gamma distribution. First draw X from the Gamma(α,β) distribution and take Y = 1/X, which corresponds to a draw from the Inverse-Gamma(α, β).<br><br>

<b>Marginal Posterior Distribution for the Normal Mean when the Variance is Unknown</b><br>
If we are not interested in inference for an unknown $\sigma^{2}$, we can integrate it out of the joint posterior in Lesson 10.2. This results in a t-distributed marginal posterior as noted at the end of the lesson. This t distribution has $\nu = 2\alpha + n$ degrees of freedom and two additional parameters, a scale $\gamma$ and a location $m^{*}$ given by<br>
the mean of the conditional posterior for μ<br>
$$m^{*} = \frac{n \bar{x} + w m}{ n + w}$$

modified scale of the updated inverse-gamma for $\sigma$<br>
$$\gamma = \sqrt{\frac{\beta + \frac{n-1}{2}s^{2} + \frac{wn}{2(w+n)}(\bar{x} - m)^{2}}{(n+w)(\alpha+n/2)}}$$

where s is sample variance<br>
$$s^{2} = \frac{1}{n-1}\sum (x_{i} - \bar{x})^{2}$$

This t distribution can be used to create a credible interval for μ by multiplying theappropriate quantiles of the standard t distribution by the scale γ and adding the location $m^{*}$.<br><br>

Example: Suppose we have normal data with unknown mean μ and variance $\sigma^{2}$. We use the model from Lesson 10.2 with m=0, w=0.1, α=3/2, and β=1. The data are n=20 independent observations with x ̄ = 1.2 and $s^{2}$ = 0.7. Then we have<br>
$$
\begin{aligned}
& \sigma^{2} \mid x\sim \text{Inverse-Gamma}(11.5, 7.72) \\
& \mu \mid \sigma^{2}, x \sim N(1.19, \frac{\sigma^{2}}{20.1}) \\
& m^{*} = 1.19 \\
& \gamma = 0.183
\end{aligned}
$$

and μ | x is distributed t with 23 degrees of freedom, location 1.19 and scale 0.183. To produce a 95% equal-tailed credible interval for μ, we first need the 0.025 and 0.975 quantiles of the standard t distribution with 23 degrees of freedom. These are −2.07 and 2.07. The 95% credible interval is then $m^{*}$ ± $\gamma$(2.07) = 1.19 ± 0.183(2.07) = 1.19 ± 0.38.<br><br>
</p>

### 1.11 Alternative priors
#### 1.11.1 Non-informative priors
<p align="justify">
Inproper prior: not a legal PDF. For example Beta(0, 0)<br>
$$f(\theta) \sim \theta^{-1}(1-\theta)^{-1}$$

Posterior<br>
$$f(\theta \mid y) \propto \theta^{y-1} (1 - \theta)^{n - y -1} \sim \text{Beta}(y, n-y)$$

Posterior mean<br>
$$\frac{y}{n} = \hat{\theta_{}}$$

This agrees with frequentist paradigm.<br><br>

<b>What does it mean for a prior to be improper?</b><br>
A. The resulting prior predictive distribution is unlikely to have produced the data.<br>
B. It's use will never result in a posterior distribution which integrates (or sums) to 1.<br>
C. Its density (PDF) can take on negative values.<br>
D. It does not integrate (or sum) to 1.<br><br>

<b>Answer</b>: D.<br><br>

Consider a variable $Y_{i} \sim N(\mu, \sigma^{2})$, vague prior is like<br>
$$\mu \sim N(0, 10000000^{2})$$

So, the density can be reagarded constant<br>
$$f(\mu) \propto 1$$

Posterior<br>
$$
\begin{aligned}
& f(\mu \mid y) \propto f(y \mid \mu) f(\mu) \propto e^{-\frac{1}{2\sigma^{2}} \sum (y_{i} - \mu)^{2}} (1) \propto e^{-\frac{1}{2 \frac{\sigma^{2}}{n}} (\mu - \bar{y})^{2}} \\
& \mu \mid y \sim N(\bar{y}, \frac{\sigma^{2}}{n})
\end{aligned}
$$

If $\sigma$ is unknown<br>
$$f(\sigma^{2}) \propto \frac{1}{\sigma^{2}}$$

which is a Inverse-Gamma(0, 0) for prior<br><br>

Posterior<br>
$$\sigma^{2} \mid y \sim \Gamma^{-1}(\frac{n-1}{2}, \frac{1}{2} \sum (y-{i} - \bar{y})^{2})$$

In simple models, non-informative priors often produce posterior mean estimates that are equivalent to the common frequentist/MLE estimates. <b>Why might we still use this Bayesian approach?</b><br>
A. The MLE is usually more difficult to compute than a posterior mean.<br>
B. In addition to the point estimate, we have a posterior distribution for the parameter which allows us to calculate posterior probabilities and credible intervals.<br>
C. Posterior mean estimates are on average 50% closer to the true parameter value than the corresponding MLE estimate. Thus, the MLE is not a trustworthy estimator.<br>
D. The posterior mean is invariant to the choice of prior.<br><br>

<b>Answer</b>: B.<br><br>
</p>

#### 1.11.2 Jeffreys prior
<p align="justify">
Jeffreys is defined as a square root of fisher information<br>
$$f(\theta) \propto \sqrt{I(\theta)}$$

For the data from normal distribution $Y_{i} \sim N(\mu, \sigma^{2}$<br>
Jeffreys for $\mu$ is uniform<br>
$$f(\mu) \propto 1$$

Jeffreys for $\sigma$<br>
$$f(\sigma^{2}) \propto \frac{1}{\sigma^{2}}$$

For the data from Bernoulli or Binomial distribution $Y_{i} \sim B(\theta)$, Jeffreys is<br>
$$f(\theta) \propto \theta^{-\frac{1}{2}} (1 - \theta)^{-\frac{1}{2}} \sim \text{Beta}(\frac{1}{2}, \frac{1}{2})$$

Jeffreys priors are "transformation invariant" in the sense that if we calculate the Jeffreys prior for θ and then reparameterize to use ϕ=g(θ), we get the same result as if we had first reparameterized and then found the Jeffrey's prior for ϕ. <b>Why might this property be desirable?</b><br>
A. No matter how we parameterize the problem, the Jeffreys prior contains the least possible prior "information."<br>
B. Different investigators might parameterize a problem in different ways. Using the Jeffreys prior ensures that they both obtain the same answer.<br>
C. The Jefferys prior is uniform on all scales.<br><br>

<b>Answer</b>: B.<br><br>
</p>

#### 1.11.3 Quiz
<p align="justify">
<b>1.</b><br>
Suppose we flip a coin five times to estimate θ, the probability of obtaining heads. We use a Bernoulli likelihood for the data and a non-informative (and improper) Beta(0,0) prior for θ. We observe the following sequence: (H, H, H, T, H).Because we observed at least one H and at least one T, the posterior is proper. What is the posterior distribution for θ?<br>
A. Beta(1.5, 4.5)<br>
B. Beta(1,4)<br>
C. Beta(4,1)<br>
D. Beta(4.5, 1.5)<br>
E. Beta(5,2)<br>
F. Beta(2,5)<br><br>

<b>Answer</b>: D.<br><br>

<b>2.</b><br>
Continuing the previous question, what is the posterior mean for θ? Round your answer to one decimal place.<br><br>

<b>Answer</b>: 0.80.<br><br>

<b>3. Consider again the thermometer calibration problem from Lesson 10.</b><br>
Assume a normal likelihood with unknown mean θ and known variance $\sigma^{2}$ = 0.25. Now use the non-informative (and improper) flat prior for θ across all real numbers. This is equivalent to a conjugate normal prior with variance equal to ∞. You collect the following n=5 measurements: (94.6, 95.4, 96.2, 94.9, 95.9). What is the posterior distribution for θ?<br>
A. N(95.4,0.05)<br>
B. N(96.0, $0.25^{2}$)<br>
C. N(95.4, 0.25)<br>
D. N(96.0, $0.05$^{2})<br><br>

<b>Answer</b>: A.<br><br>

<b>4.</b><br>
Which of the following graphs shows the Jeffreys prior for a Bernoulli/binomial success probability p? Hint: The Jeffreys prior in this case is Beta(1/2, 1/2).<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/1_11_3_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: B.<br><br>

<b>5.</b><br>
Scientist A studies the probability of a certain outcome of an experiment and calls it θ. To be non-informative, he assumes a Uniform(0,1) prior for θ. Scientist B studies the same outcome of the same experiment using the same data, but wishes to model the odds $\theta$/(1 - $\theta$). Scientiest B places a uniform distribution on ϕ. If she reports her inferences in terms of the probability θ, will they be equivalent to the inferences made by Scientist A?<br>
A. Yes, they both used uniform priors.<br>
B. Yes, they used the Jeffreys prior.<br>
C. No, they are using different parameterizations.<br>
D. No, they did not use the Jeffreys prior.<br><br>

<b>Answer</b>: D.<br><br>
</p>

#### 1.11.4 Supplementary material for Lesson 11
<p align="justify">
<b>Fisher Information</b><br>
The Fisher information (for one parameter) is defined as<br>
$$(\theta) = E[(\frac{d}{d\theta} \log (f(X \mid \theta)))^{2}]$$

where the expectation is taken with respect to X which has PDF f(x | θ). This quantity is useful in obtaining estimators for θ with good properties, such as low variance. It is also the basis for the Jeffreys prior.<br><br>

For example, let X | $\theta$ $\sim$ N(0, 1). Then we have<br>
$$
\begin{aligned}
& f(x \mid \theta) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2} (x - \theta)^{2}} \\
& \log(f(x \mid \theta)) = -\frac{1}{2} \log(2 \pi) - \frac{1}{2}(x - \theta)^{2} \\
& \frac{d}{d\theta} \log(f(x \mid \theta)) = -\frac{2}{2} (x - \theta) (-1) = x - \theta \\
& (\frac{d}{d\theta} \log(f(x \mid \theta)))^{2} = (x - \theta)^{2}
\end{aligned}
$$

So,<br>
$$I(\theta) = E[(X - \theta)^{2}] = \text{Var}(X) = 1$$
<br>
</p>

### 1.12 Linear regression
#### 1.12.1 Background for Lesson 12
<p align="justify">
<b>Brief Review of Regression</b><br>
Recall that linear regression is a model for predicting a response or dependent variable (Y , also called an output) from one or more covariates or independent variables (X, also called explanatory variables, inputs, or features). For a given value of a single x, the expected value of y is<br>
$$E[y] = \beta_{0} + \beta_{1}x$$

or we could say that Y $\sim$ N($\beta_{0}$ + $\beta_{1}x$, $\sigma^{2}$). FOr data ($x_{1}$, $y_{1}$), ..., ($x_{n}$, $y_{n}$), the fitted values for
the coefficients, $\hat{\beta_{0}}$ and $\hat{\beta_{1}}$ are those that minimize the sum of squared errors<br>
$$\sum_{i=1}^{n} (y_{i} - \hat{y_{i}})^{2}$$

where the predicted values for the response are<br>
$$\hat{y_{}} = \hat{\beta_{0}} + \hat{\beta_{1}} x$$

We can get these values from R or Excel. These fitted coefficients give the least-squares line for the data. This model extends to multiple covariates, with one βj for each of the k covariates:<br>
$$E[y_{i}] = \beta_{0} + \beta_{1} x_{i1} + ... + \beta_{k} x_{ik}$$

Optionally, we can represent the multivariate case using vector-matrix notation.<br><br>

<b>Conjugate Modeling</b><br>
In the Bayesian framework, we treat the β parameters as unknown, put a prior on them, and then find the posterior. We might treat $\sigma^{2}$ as fixed and known, or we might treat it as unknown and also put a prior on it. Because the underlying assumption of a regression model is that the errors are independent and identically normally distributed with mean zero and variance σ2, this defines a normal likelihood.<br><br>

<b>$\sigma^{2}$ is known</b><br>
Sometimes we may know the value of the error variance $\sigma^{2}$. This simplifies the calculations. The conjugate prior for the β’s is a normal prior. In practice, people typically use a non- informative prior, i.e., the limit as the variance of the normal prior goes to infinity, which is a completely flat prior, and is also the Jeffreys prior. Using this prior gives a posterior distribution for β which has the same mean as the standard least-squares estimates. If we are only estimating β and treating $\sigma^{2}$ as known, then the posterior for β is a (multivariate) normal distribution. If we just have a single covariate, then the posterior for the slope is<br>
$$\beta_{1} \mid y \sim N(\frac{\sum_{i=1}^{n} (x_{i} - \bar{x})(y_{i} - \bar{y})}{\sum_{i=1}^{n} (x_{i} - \bar{x})^{2}}, \frac{\sigma^{2}}{\sum_{i=1}^{n} (x_{i} - \bar{x})^{2}})$$

If we have multiple covariates, then using matrix-vector notation, the posterior for the vector of coefficients is<br>
$$\beta \mid y \sim N((X^{T}X)^{-1}X^{T}y, (X^{T}X)^{-1} \sigma^{2})$$

where X denotes the design matrix and $X^{T}$ is the transpose of X. The intercept is typically included in X as a column of 1’s. Using an improper prior requires us to have at least as many data points as we have parameters to ensure the the posterior is proper.<br><br>

<b>$\sigma^{2}$ is unknown</b><br>
If we treat both β and σ2 as unknown, the standard prior is the non-informative Jeffreys prior,<br>
$$f(\beta, \sigma^{2}) \propto \frac{1}{\sigma^{2}}$$

Again, the posterior mean for β will be the same as the standard least-squares estimates. The posterior for β conditional on $\sigma^{2}$ is the same normal distribution as when $\sigma^{2}$ is known, but the marginal posterior distribution for β, with $\sigma^{2}$ integrated out is a t distribution, analogous to the t tests for significance in standard linear regression. The posterior t distribution has mean $(X^{T}X)^{-1}X^{T}y$ and scale matrix (related to the variance matrix) $s^{2} (X^{T}X)^{-1}$, where s is<br>
$$s^{2} = \sum_{i=1}^{n} \frac{(y_{i} - \hat{y_{i}})^{2}}{n - k - 1}$$

The posterior distribution for $\sigma^{2}$ is an inverse gamma distribution<br>
$$\sigma^{2} \mid y \sim IG(\frac{n-k-1}{2}, \frac{n-k-1}{2} s^{2})$$

In the simple linear regression case (single variable), the marginal posterior for β is a t distribution with mean<br>
$$\frac{\sum_{i=1}^{n} (x_{i} - \bar{x}) (y_{i} - \bar{y})}{\sum_{i=1}^{n} (x_{i} - \bar{x})^{2}}$$

and scale<br>
$$\frac{s^{2}}{\sum_{i=1}^{n} (x_{i} - \bar{x})^{2}}$$

If we are trying to predict a new observation at a specified input $x^{*}$, that predicted value has a marginal posterior predictive distribution that is a t-distribution, with mean $\hat{y_{}} = \hat{\beta_{0}} + \hat{\beta_{1}} x^{*}$ and scale<br>
$$\sqrt{1 + \frac{1}{n} + \frac{(x^{*} - \bar{x}^{2})}{(n - 1)s_{x}^{2}}}$$

is the residual standard error of the regression, which can be found easily in R or Excel. $s_{x}^{2}$ is the sample variance of x. Recall that the predictive distribution for a new observation has more variability than the posterior distribution for $\hat{y_{}}$, because individual observations are more variable than the mean.<br><br>
</p>

#### 1.12.2 Linear regression in R
{% highlight R %}
http://www.randomservices.org/random/data/Challenger2.txt
# 23 previous space shuttle launches before the Challenger disaster
# T is the temperature in Fahrenheit, I is the O-ring damage index

oring=read.table("http://www.randomservices.org/random/data/Challenger2.txt",
                  header=T)
attach(oring)
#note: masking T=TRUE

plot(T,I)

oring.lm=lm(I~T)
summary(oring.lm)

# add fitted line to scatterplot
lines(T,fitted(oring.lm))
# 95% posterior interval for the slope
-0.24337 - 0.06349*qt(.975,21)
-0.24337 + 0.06349*qt(.975,21)
# note that these are the same as the frequentist confidence intervals

# the Challenger launch was at 31 degrees Fahrenheit
# how much o-ring damage would we predict?
# y-hat
18.36508-0.24337*31
coef(oring.lm)
coef(oring.lm)[1] + coef(oring.lm)[2]*31

# posterior prediction interval (same as frequentist)
predict(oring.lm,data.frame(T=31),interval="predict")  
10.82052-2.102*qt(.975,21)*sqrt(1+1/23+((31-mean(T))^2/22/var(T)))

# posterior probability that damage index is greater than zero
1-pt((0-10.82052)/(2.102*sqrt(1+1/23+((31-mean(T))^2/22/var(T)))),21)


http://www.randomservices.org/random/data/Galton.txt
# Galton's seminal data on predicting the height of children from the 
# heights of the parents, all in inches

heights=read.table("http://www.randomservices.org/random/data/Galton.txt",
                    header=T)
attach(heights)
names(heights)

pairs(heights)
summary(lm(Height~Father+Mother+Gender+Kids))
summary(lm(Height~Father+Mother+Gender))
heights.lm=lm(Height~Father+Mother+Gender)

# each extra inch taller a father is is correlated with 0.4 inch extra
  height in the child
# each extra inch taller a mother is is correlated with 0.3 inch extra
  height in the child
# a male child is on average 5.2 inches taller than a female child
# 95% posterior interval for the the difference in height by gender
5.226 - 0.144*qt(.975,894)
5.226 + 0.144*qt(.975,894)

# posterior prediction interval (same as frequentist)
predict(heights.lm,data.frame(Father=68,Mother=64,Gender="M"),
        interval="predict")
predict(heights.lm,data.frame(Father=68,Mother=64,Gender="F"),
        interval="predict")
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 1.12.3 Linear regression in Excel (Analysis ToolPak)
{% highlight Excel %}
http://www.randomservices.org/random/data/Challenger2.txt
# 23 previous space shuttle launches before the Challenger disaster
# T is the temperature in Fahrenheit, I is the O-ring damage index

<cut and paste into spreadsheet>
<insert-chart-X Y (Scatter)>
<select series-add chart element-add trendline-linear>

<data-data analysis-regression>
 <Y Range, X Range, Output Range>

# 95% posterior interval for the slope
# given in table, or can compute by hand
= -0.24337 - 0.06349*T.INV(.975,21)
= -0.24337 + 0.06349*T.INV(.975,21)
# note that these are the same as the frequentist confidence intervals

# the Challenger launch was at 31 degrees Fahrenheit
# how much o-ring damage would we predict?
# y-hat
= 18.36508-0.24337*31

# posterior prediction interval (same as frequentist)
=10.82-2.102*T.INV(.975,21)*SQRT(1+1/23+
((31-AVERAGE(A2:A24))^2/22/VARA(A2:A24)))
=10.82+2.102*T.INV(.975,21)*SQRT(1+1/23+
((31-AVERAGE(A2:A24))^2/22/VARA(A2:A24)))

# posterior probability that damage index is greater than zero
=1-T.DIST((0-10.82052)/(2.102*SQRT(1+1/23+
((31-AVERAGE(A2:A24))^2/22/VARA(A2:A24)))),21,TRUE)



http://www.randomservices.org/random/data/Galton.txt
# Galton's seminal data on predicting the height of children from the 
# heights of the parents, all in inches

<cut and paste into spreadsheet>
<copy Father and Mother columns>
<make GenderM column of 1's and 0's>
=IF(D2="M",1,0)
<copy and paste to fill column>

<data-data analysis-regression>
 <Y Range, X Range, Output Range>
   <X Range=Kids+Father+Mother+GenderM>
   <X Range=Father+Mother+GenderM>

# each extra inch taller a father is is correlated with 0.4 inch extra
  height in the child
# each extra inch taller a mother is is correlated with 0.3 inch extra
  height in the child
# a male child is on average 5.2 inches taller than a female child
# 95% posterior interval for the the difference in height by gender
#  is given in the table:  (4.94, 5.51)
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 1.12.4 Linear regression in Excel (StatPlus by AnalystSoft)
{% highlight Excel %}
http://www.randomservices.org/random/data/Challenger2.txt
# 23 previous space shuttle launches before the Challenger disaster
# T is the temperature in Fahrenheit, I is the O-ring damage index

<cut and paste into spreadsheet>

<regression-multiple linear regression>
 <dependent=I, Independent=T>

# 95% posterior interval for the slope
# given in table, or can compute by hand
= -0.24337 - 0.06349*T.INV(.975,21)
= -0.24337 + 0.06349*T.INV(.975,21)
# note that these are the same as the frequentist confidence intervals

# the Challenger launch was at 31 degrees Fahrenheit
# how much o-ring damage would we predict?
# y-hat
= 18.36508-0.24337*31

# posterior prediction interval (same as frequentist)
=10.82-2.102*T.INV(.975,21)*SQRT(1+1/23+
((31-AVERAGE(A2:A24))^2/22/VARA(A2:A24)))
=10.82+2.102*T.INV(.975,21)*SQRT(1+1/23+
((31-AVERAGE(A2:A24))^2/22/VARA(A2:A24)))

# posterior probability that damage index is greater than zero
=1-T.DIST((0-10.82052)/(2.102*SQRT(1+1/23+
((31-AVERAGE(A2:A24))^2/22/VARA(A2:A24)))),21,TRUE)
=T.DIST(10.82052/(2.102*SQRT(1+1/23+
((31-AVERAGE(A2:A24))^2/22/VARA(A2:A24)))),21,TRUE)



http://www.randomservices.org/random/data/Galton.txt
# Galton's seminal data on predicting the height of children from the
# heights of the parents, all in inches

<cut and paste into spreadsheet>
<make GenderM column of 1's and 0's>
=IF(D2="M",1,0)
<copy and paste to fill column>

<regression-multiple linear regression>
 <dependent=Height, Independent=Father+Mother+Kids+GenderM>
 <dependent=Height, Independent=Father+Mother+GenderM>

# each extra inch taller a father is is correlated with 0.4 inch extra
  height in the child
# each extra inch taller a mother is is correlated with 0.3 inch extra
  height in the child
# a male child is on average 5.2 inches taller than a female child
# 95% posterior interval for the the difference in height by gender
#  is given in the table:  (4.94, 5.51)
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 1.12.6 Quiz
<p align="justify">
<b>1. For Questions 1-6, consider the following:</b><br>
The data found at http://www.stat.ufl.edu/~winner/data/pgalpga2008.dat consist of season statistics for individual golfers on the United States LPGA and PGA tours. The first column reports each player's average driving distance in yards. The second column reports the percentage of the player's drives that finish in the fairway, measuring their accuracy. The third and final column has a 1 to denote a female golfer (on the LPGA tour), and a 2 to denote male golfer (on the PGA tour).<br><br>

Load these data into R or Excel. In Excel, once you paste the data into a new worksheet, you may need to separate the data into columns using the "Text to Columns" feature under the "Data" menu.<br><br>

If you wish to separate the LPGA and PGA data, one way in R is to use the subset function:<br>
</p>
{% highlight R %}
datF <- subset(dat, FM==1, select=1:2)
{% endhighlight %}
<p align="justify">
where "dat" is the name of the original data set (replace "dat" with whatever you named this data set), "FM" is the name of the third column (replace "FM" with whatever you named this column), and select=1:2 means to include columns 1 and 2 in the new data set "datF". Create two scatter plots with average drive distance on the x-axis and percent accuracy on the y-axis, one for female golfers and one for male golfers. What do you observe about the relationship between these two variables?<br>
A. Drive distance and accuracy are positively correlated; greater distances are associated with greater accuracy.<br>
B. Drive distance and accuracy are negatively correlated; greater distances are associated with less accuracy.<br>
C. There is no association between driving distance and accuracy.<br>
<b>Answer</b>: B.<br><br>

<b>2.</b><br>
Fit a linear regression model to the female golfer data only with drive distance as the explanatory variable x and accuracy as the response variable y. Use the standard reference (non-informative) prior. Recall that in a linear regression, we are modeling E(y) = $b_{0}$ + $b_{1}$x. In this particular model, the intercept term is not interpretable, as we would not expect to see a 0-yard drive (but it is still necessary). Predictions should generally be made only within the range of the observed data. Report the posterior mean estimate of the slope parameter b relating drive distance to accuracy. Round your answer to two decimal places.<br>
<b>Answer</b>: -0.26.<br><br>

<b>3.</b><br>
The posterior mean estimate of the slope from Question 2 is about five standard errors below 0. Hence, the posterior probability that this slope is negative is near 1. Suppose the estimate is b. How do we interpret this value?<br>
A. If x is the driving distance, we expect the percentage accuracy to be 100bx.<br>
B. If x is the driving distance, we expect the percentage accuracy to be bx.<br>
C. For each additional yard of driving distance, we expect to see an increase in percentage accuracy of ∣b∣.<br>
D. For each additional yard of driving distance, we expect to see a decrease in percentage accuracy of ∣b∣.<br>
<b>Answer</b>: D.<br><br>

<b>4.</b><br>
Use the posterior mean estimates of the model coefficients to obtain a posterior predictive mean estimate of driving accuracy for a new female golfer whose average driving distance is x=260 yards. Round your answer to one decimal place.<br>
<b>Answer</b>: 64.21.<br><br>

<b>5.</b><br>
Which of the following gives a 95% posterior predictive interval for the driving accuracy of a new female golfer whose average driving distance is x=260 yards? Hint: Modify the code provided with this lesson under "prediction interval."<br>
A. (62.8, 65.6)<br>
B. (63.0, 65.4)<br>
C. (53.7, 74.7)<br>
D. (55.4, 73.0)<br>
<b>Answer</b>: C.<br><br>

<b>6*.</b><br>
What is the correct interpretation of the interval found in Question 5?<br>
A. If we select a new female golfer who averages 260 yards per drive, we are 95% confident that the posterior mean for her accuracy would be in the interval.<br>
B. For all female golfers who average 260 yards per drive, we are 95% confident that all their driving accuracies will be in the interval.<br>
C. If we select a new female golfer who averages 260 yards per drive, our probability that her driving accuracy will be in the interval is .95.<br>
D. For all female golfers who average 260 yards per drive, our probability is .95 that the mean of their driving accuracy is in the interval.<br>
<b>Answer</b>: C.<br><br>
</p>

### 1.13 Honor Quiz
#### 1.13.1 Probability and Bayes' theorem
<p align="justify">
<b>1.</b><br>
Which of the following (possibly more than one) must be true if random variable X is continuous with PDF f(x)?<br>
A. $\lim_{x \rightarrow \infty} f(x) = \infty$<br>
B. X $\geq$ 0 always<br>
C. $\int_{-\infty}^{\infty} f(x) dx = 1$<br>
D. f(x) is an increasing function of x<br>
E. f(x) $\geq$ 0 always<br>
F. f(x) is a continuous function<br>
<b>Answer</b>: C, E.<br><br>

<b>2.</b><br>
If X $\sim$ Exp(3), what is the value of P(X > 1/3)? Round your answer to two decimal places.<br>
<b>Answer</b>: 0.37.<br><br>

<b>3.</b><br>
Suppose X $\sim$ Uniform(0, 2) and Y $\sim$ Uniform(8, 10). What is the value of E(4X + Y)?<br>
<b>Answer</b>: 13.<br><br>

<b>4. For Questions 4-7, consider the following:</b><br>
Suppose X $\sim$ N(1, $5^{2}$) and Y $\sim$ N(-2, $3^{2}$) and that X and Y are independent. We have Z = X + Y $\sim$ N($\mu$, $\sigma^{2}$) because the sum of normal random variable also follows a normal distribution. What is the value of $\mu$?<br>
<b>Answer</b>: -1.<br><br>

<b>5.</b><br>
What is the value of $\sigma^{2}$? Hint: If two random variables are independent, the variance of their sum is the sum of their variances.<br>
<b>Answer</b>: 34.<br><br>

<b>6.</b><br>
If random variables X and Y are not independent, we still have<br>
$$E(X + Y) = E(X) + E(Y)$$

but now<br>
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y), \quad \text{Cov}(X, Y) = E[(X - E[X]) (Y - E[Y])]$$

$\text{Cov}(X, Y)$ is called the covariance between X and Y.<br><br>

A convient formula for calculating variance was given<br>
$$\text{Var}(X) = E[(X - E[X])^{2}] = E[X^{2}] - (E[X])^{2}$$

Which of the following is an analogous expression for the covariance of X and Y?<br>
$$
\begin{aligned}
& \text{A.} \quad (E[X^{2}] - (E[X])^{2}) \cdot (E[Y^{2}] - (E[Y])^{2}) \\
\\
& \text{B.} \quad E[Y^{2}] - (E[Y])^{2} \\
\\
& \text{C.} \quad E[X^{2}] - (E[X])^{2} + E[Y^{2}] - (E[Y])^{2} \\
\\
& \text{D.} \quad E(XY) - E(X)E(Y)
\end{aligned}
$$
<b>Answer</b>: D.<br><br>

<b>7.</b><br>
Consider again X $\sim$ N(1, $5^{2}$) and Y $\sim$ N(-2, $3^{2}$) but that X and Y are dependent. Z = X + Y is still normally distributed with the same mean found in Question 4. What is the variance of Z if E(XY) = -5?<br>
<b>Answer</b>: 28.<br><br>

<b>8. Free point:</b><br>
1) Use the definition of conditional probability to show that for events A and B, we have<br>
$$P(A \cap B) = P(B \mid A) P(A) = P(A \cap B) P(B)$$

2) Show that the two expressions for independence are equivalent<br>
$$
\begin{aligned}
& P(A \mid B) = P(A) \\
& P(A \cap B) = P(A) P(B)
\end{aligned}
$$
<b>Answer</b>: 1) 2).<br><br>
</p>

#### 1.13.2 Statistical inference
<p align="justify">
<b>1.</b><br>
Although the likelihood function is not always a product of $f(y_{i} \mid \theta)$ for i = 1, 2, ..., n, this product form is convenient mathematically. What assumption about the observations y allows us to multiply their individual likelihood components?<br>
<b>Answer</b>: Independent.<br><br>

<b>2.</b><br>
One nice property of MLEs is that they are transformation invariant. That is, if $\hat{\theta_{}}$ is the MLE for θ, then the MLE for g($\theta$) is g($\hat{\theta_{}}$) for any function g($\cdot$). Suppose you conduct 25 Bernoulli trials and observe 10 successes. What is the MLE for the odds of success? Round your answer to two decimal places.<br>
<b>Answer</b>: 0.67.<br>
$$\text{Odd} = \frac{P}{1 - P}$$

<b>3. For Questions 3-4</b><br>
recall the scenario from Lesson 5 in which your brother brings you a coin which may be fair (probability of heads 0.5) or loaded (probability of heads 0.7). Another sibling wants to place bets on whether the coin is loaded. If the coin is actually loaded, she will pay you 1. If it is not loaded, you will pay her z. Using your prior probability of 0.6 that the coin is loaded and assuming a fair game, determine the amount z that would make the bet fair (with prior expectation $0). Round your answer to one decimal place.<br>
<b>Answer</b>: 1.5.<br>
$$E(\theta) = 0.6 \times 1 + 0.4 \times (-z) = 0$$

<b>4.</b><br>
Before taking the bet, you agree to flip the coin once. It lands heads. Your sister argues that this is evidence for the loaded coin (in which case she pays you $1) and demands you increase z to 2. Should you accept this new bet? Base your answer on your updated (posterior) probability that the coin is loaded.<br>
A. Yes, your posterior expected payoff is now less than 0.<br>
B. Yes, your posterior expected payoff is now greater than 0.<br>
C. No, your posterior expected payoff is now less than 0.<br>
D. No, your posterior expected payoff is now greater than 0.<br>
<b>Answer</b>: B.<br>
Posterior probability after observing 1 head<br>
$$
\begin{aligned}
f(\theta \mid X = 1) & = \frac{f(X=1 \mid \theta) f(\theta)}{\sum_{\theta} f(X = 1 \mid \theta) f(\theta)} \\
& = \frac{(0.7)^{1} (0.3)^{0} 0.6 \mathbf{I}_{\theta = \text{fair}} + (0.5)^{1}(0.5)^{0} (0.4) \mathbf{I}_{\theta = \text{loaded}}}{(0.7)^{1} (0.3)^{0} (0.6) + (0.5)^{1} (0.5)^{0} (0.4)} \\
& \approx 0.677
\end{aligned}
$$
$$E(\theta) = 0.677 \times 1 - 0.323 \times 2 = 0.031 > 0$$

The fair amount would have been z=2.1.<br><br>
</p>

#### 1.13.3 Priors and models for discrete data
<p align="justify">
<b>1.</b><br>
Identify which of the following conditions (possibly more than one) must be true for the sum of n Bernoulli random variables (with success probability p) to follow a binomial distribution.<br>
A. each Bernoulli random variable is independent of all others<br>
B. the sum must be greater than zero<br>
C. p must be less than .5<br>
D. p must be the same for each of the Bernoulli random variables<br>
E. the sum must exceed n<br>
<b>Answer</b>: A, D.<br><br>

<b>2. For Questions 2-4, consider the following:</b><br>
In Lesson 6.3 we found the prior predictive distribution for a Bernoulli trial under a uniform prior on the success probability θ. We now derive the prior predictive distribution when the prior is any conjugate beta distribution. There are two straightforward ways to do this. The first approach is the same as in the lesson. The marginal distribution of y is<br>
$$f(y) = \int_{0}^{1} f(y \mid \theta) f(\theta) d\theta$$

Now $f(\theta)$ is a beta PDF, but the same principles apply: we can move constants out of the integral and find a new normalizing constant to make the integral evaluate to 1. Another approach is to notice that we can write Bayes' theorem as<br>
$$f(\theta \mid y) = \frac{f(y \mid \theta) f(\theta)}{f(y)}$$

If we multiply both sides by f(y) and divide both sides by $f(\theta \mid y)$, then we get<br>
$$f(y) = \frac{f(y \mid \theta) f(\theta)}{f(\theta \mid y)}$$

where $f(\theta)$ is the beta prior PDF and $f(\theta \mid y)$ is the updated beta posterior PDF. Both approaches yield the same answer. What is the prior predictive distribution f(y) for this model when the prior for $\theta$ is Beta(a, b)?<br>
$$
\begin{aligned}
& \text{A.} \quad f(y) = \frac{\Gamma(a + b)}{\Gamma(a) \Gamma(b)} \theta^{a-1} (1 - \theta)^{b-1}, \quad \text{for } y = 0, 1 \\
\\
& \text{B.} \quad f(y) = \frac{\Gamma(a + b)}{\Gamma(a + b + 1)} \cdot \frac{\Gamma(a + y)}{\Gamma(a)} \cdot \frac{\Gamma(b + 1 - y)}{\Gamma(b)}, \quad \text{for } y = 0, 1 \\
\\
& \text{C.} \quad f(y) = \frac{\Gamma(a + y)}{\Gamma(a)} \cdot \frac{\Gamma(b + 1 - y)}{\Gamma(b)}, \quad \text{for } y = 0, 1 \\
\\
& \text{D.} \quad f(y) = \frac{\Gamma(a + b + 1)}{\Gamma(a + y) \Gamma(b + 1 - y)} \theta^{a + y -1} (1 - \theta)^{b + 1 - y}, \quad \text{for } y = 0, 1\\
\\
& \text{E.} \quad f(y) = \frac{\Gamma(a + b)}{\Gamma(a + b + 1)} \theta^{y} (1 - \theta)^{1 - y}, \quad \text{for } y = 0, 1
\end{aligned}
$$
<b>Answer</b>: B.<br><br>

<b>3.</b><br>
Now suppose the prior for θ is Beta(2,2). What is the prior predictive probability that $y^{*}$ = 1 for a new observation $y^{*}$? Round your answer to one decimal place.<br>
<b>Answer</b>: 0.5.<br><br>

<b>4.</b><br>
After specifying our Beta(2,2) prior for θ, we observe 10 Bernoulli trials, 3 of which are successes. What is the posterior predictive probability that $y^{*}$ = 1 for the next (11th) observation $y^{*}$? Round your answer to two decimal places.<br>
<b>Answer</b>: 0.36.<br>
After 10 trials, prior becomes Beta(5, 9), with this new prior, we can calculate a predicted prior for 11th data<br>
$$
\begin{aligned}
f(y) & = \int_{0}^{1} f(y \mid \theta) f(\theta) d\theta \\
& = \int_{0}^{1} \theta^{y} (1-\theta)^{1-y} \frac{\Gamma(a+b)}{\Gamma(a) \Gamma(b)} \theta^{a-1} (1-\theta)^{b-1} d\theta \\
& = \frac{\Gamma(a + b)}{\Gamma(a + b + 1)} \cdot \frac{\Gamma(a + y)}{\Gamma(a)} \cdot \frac{\Gamma(b + 1 - y)}{\Gamma(b)}, \quad \text{for } y = 0, 1
\end{aligned}
$$
</p>

#### 1.13.4 Models for continuous data
<p align="justify">
<b>1. Consider again the golf data from the regression quiz for Questions 1-4.</b><br>
The data are found at http://www.stat.ufl.edu/~winner/data/pgalpga2008.dat and consist of season statistics for individual golfers on the United States LPGA and PGA tours. The first column reports each player's average driving distance in yards. The second column reports the percentage of the player's drives that finish in the fairway, measuring their accuracy. The third and final column has a 1 to denote a female golfer (on the LPGA tour), and a 2 to denote male golfer (on the PGA tour).<br><br>

Now consider a multiple regression on the full data set, including both female and male golfers. Modify the third variable to be a 0 if the golfer is female and 1 if the golfer is male and fit the following regression:<br>
$$E(y) = b_{0} + b_{1} x + b_{2} x_{2}$$

where $x_{1}$ is the average driving distance and $x_{2}$ is the indicator that the golfer is male. What is the posterior mean estimate of $b_{0}$? Round your answer to the nearest whole number.<br><br>

<b>Answer</b>: 147.26894 $\approx$ 147.<br>
lm(V2~V1+V3)<br><br>

<b>2.</b><br>
The posterior mean estimates of the other two coefficients are $\hat{b_{1}}$ = -0.323 and $\hat{b_{2}}$ = 8.94. What is the interpretation of $\hat{b_{1}}$?<br>
A. Holding all else constant, each additional yard of distance is associated with a 0.323 decrease in drive accuracy percentage.<br>
B. Holding all else constant, being male is associated with a 0.323 decrease in drive accuracy percentage.<br>
C. Holding all else constant, each additional yard of distance is associated with a 0.323 increase in drive accuracy percentage.<br>
D. Holding all else constant, being male is associated with a 0.323 increase in drive accuracy percentage.<br><br>

<b>Answer</b>: A.<br><br>

<b>3.</b><br>
The standard error for $b_{1}$ (which we can think of as marginal posterior standard deviation in this case) is roughly 1/10 times the magnitude of the posterior mean estimate $\hat{b_{1}}$ = -0.323. In other words, the posterior mean is more than 10 posterior standard deviations from 0. What does this suggest?<br>
A. The posterior probability that $b_{1}$ < 0 is about 0.5, suggesting no evidence for an association between driving distance and accuracy.<br>
B. The posterior probability that $b_{1}$ < 0 is very low, suggesting a negative relationship between driving distance and accuracy.<br>
C. The posterior probability that $b_{1}$ < 0 is very high, suggesting a negative relationship between driving distance and accuracy.<br><br>

<b>Answer</b>: C.<br><br>

<b>4.</b><br>
The estimated value of $b_{2}$ would typically be interpreted to mean that holding all else constant (for a fixed driving distance), golfers on the PGA tour are about 9% more accurate with their drives on average than golfers on the LPGA tour. However, if you explore the data, you will find that the PGA tour golfers' average drives are 40+ yards longer than LPGA tour golfers' average drives, and that the LPGA tour golfers are actually more accurate on average. Thus $b_{2}$, while a vital component of the model, is actually a correction for the discrepancy in driving distances. Although model fitting can be easy (especially with software), interpreting the results requires a thoughtful approach.<br><br>

It would also be prudent to check that the model fits the data well. One of the primary tools in regression analysis is the residual plot. Residuals are defined as the observed values y minus their predicted values $\hat{y_{}}$. Patterns in the plot of $\hat{y_{}}$ versus residuals, for example, can indicate an inadequacy in the model. These plots are easy to produce.<br><br>

In R:<br>
</p>
{% highlight R %}
plot(fitted(mod), residuals(mod))
{% endhighlight %}
<p align="justify">
where "mod" is the model object fitted with the lm() command.<br><br>

In Excel, residual plots are available as an output option in the regression dialogue box.<br><br>

Fit the regression and examine the residual plots. Which of the following statements most accurately describes the residual plots for this analysis?<br>
A. The residuals appear to exhibit a curved trend. There is at least one outlier (extreme observation) that we may want to investigate.<br>
B. The residuals appear to be random and lack any patterns or trends. However, there is at least one outlier (extreme observation) that we may want to investigate.<br>
C. The residuals appear to be random and lack any patterns or trends. There are no outliers (extreme observations).<br>
D. The residuals appear to be more spread apart for smaller predicted values $\hat{y_{}}$. There are no outliers (extreme observations).<br><br>

<b>Answer</b>: B.<br><br>
</p>

## 2. Bayesian Statistics: Techniques and Models
### 2.1 Statistical Modeling
#### 2.1.1 Objectives
<p align="justify">
Statistical model is a mathematical structure used to imitate and approxiamte data generating process.<br><br>

<b>Four common objectives</b><br>
1) Quantify uncertainty<br>
2) Inference<br>
3) Measures support for hypothesis<br>
4) Prediction<br><br>

<b>Q</b>: <b>Which objective does statistical modeling most share with machine learning?</b><br>
A. Quantify uncertainty<br>
B. Inference<br>
C. Measuring evidence for/against hypotheses<br>
D. Prediction<br><br>

<b>Answer</b>: D.<br><br>
</p>

#### 2.1.2 Modeling process
<p align="justify">
<b>Statistical modeling process</b><br>
1) understand the problem<br>
2) plan and collect data<br>
3) explore data<br>
4) postulate model<br>
5) fit model<br>
7) iterate<br>
8) use model<br><br>

<b>Q</b>: This entire statistical modeling process is important in both the Bayesian and frequentist paradigms. <b>In which of the following steps would the two approaches most likely differ?</b><br>
A. Explore data<br>
B. Understand the problem<br>
C. Check the model<br>
D. Fit the model<br>
<b>Answer</b>: D.<br><br>
</p>

#### 2.1.3 Quiz
<p align="justify">
<b>1.</b><br>
Which objective of statistical modeling is best illustrated by the following example? You fit a linear regression of monthly stock values for your company. You use the estimates and recent stock history to calculate a forecast of the stock's value for the next three months.<br>
A. Quantify uncertainty<br>
B. Inference<br>
C. Hypothesis testing<br>
D. Prediction<br>
<b>Answer</b>: D.<br><br>

<b>2.</b><br>
Which objective of statistical modeling is best illustrated by the following example? A biologist proposes a treatment to decrease genetic variation in plant size. She conducts an experiment and asks you (the statistician) to analyze the data to conclude whether a 10% decrease in variation has occurred.<br>
A. Quantify uncertainty<br>
B. Inference<br>
C. Hypothesis testing<br>
D. Prediction<br>
<b>Answer</b>: C.<br><br>

<b>3.</b><br>
Which objective of statistical modeling is best illustrated by the following example? The same biologist form the previous question asks you how many experiments would be necessary to have a 95% chance at detecting a 10% decrease in plant variation.<br>
A. Quantify uncertainty<br>
B. Inference<br>
C. Hypothesis testing<br>
D. Prediction<br>
<b>Answer</b>: A.<br><br>

<b>4.</b><br>
Which of the following scenarios best illustrates the statistical modeling objective of inference?<br>
A. A social scientist collects data and detects positive correlation between sleep deprivation and traffic accidents.<br>
B. A venture capitalist uses data about several companies to build a model and makes recommendations about which company to invest in next based on growth forecasts.<br>
C. A natural language processing algorithm analyzes the first four words of a sentence and provides words to complete the sentence.<br>
D. A model inputs academic performance of 1000 students and predicts which student will be valedictorian after another year of school.<br>
<b>Answer</b>: A.<br><br>

<b>5.</b><br>
Which step in the statistical modeling cycle was not followed in the following scenario? Susan gathers data recording heights of children and fits a linear regression predicting height from age. To her surprise, the model does not predict well the heights for ages 14-17 (because the growth rate changes with age), both for children included in the original data as well as other children outside the model training data.<br>
A. Plan and properly collect relevant data<br>
B. Use the model<br>
C. Fit the model<br>
D. Explore the data<br>
<b>Answer</b>: D.<br><br>

<b>6.</b><br>
Which of the following is a possible consequence of failure to plan and properly collect relevant data?<br>
A. You will not produce enough data to make conclusions with a sufficient degree of confidence.<br>
B. Your selected model will not be able to fit the data.<br>
C. You may not be able to visually explore the data.<br>
D. Your analysis may produce incomplete or misleading results.<br>
<b>Answer</b>: D.<br><br>

<b>7. For Questions 6 and 7, consider the following:</b><br>
Xie operates a bakery and wants to use a statistical model to determine how many loaves of bread he should bake each day in preparation for weekday lunch hours. He decides to fit a Poisson model to count the demand for bread. He selects two weeks which have typical business, and for those two weeks, counts how many loaves are sold during the lunch hour each day. He fits the model, which estimates that the daily demand averages 22.3 loaves. Over the next month, Xie bakes 23 loaves each day, but is disappointed to find that on most days he has excess bread and on a few days (usually Mondays), he runs out of loaves early. Which of the following steps of the modeling process did Xie skip?<br>
A. Understand the problem<br>
B. Postulate a model<br>
C. Fit the model<br>
D. Check the model and iterate<br>
E. Use the model<br>
<b>Answer</b>: D.<br><br>

<b>8.</b><br>
What might you recommend Xie do next to fix this omission and improve his predictive performance?<br>
A. Abandon his statistical modeling initiative.<br>
B. Collect three more weeks of data from his bakery and other bakeries throughout the city. Re-fit the same model to the extra data and follow the results based on more data.<br>
C. Plot daily demand and model predictions against the day of the week to check for patterns that may account for the extra variability. Fit and check a new model which accounts for this.<br>
D. Trust the current model and continue to produce 23 loaves daily, since in the long-run average, his error is zero.<br>
<b>Answer</b>: C.<br><br>
</p>

### 2.2 Bayesian Modeling
#### 2.2.1 Components of Bayesian models
<p align="justify">
Suppose data is 15 heights of men<br>
We assume the data follows a normal distribution<br>
$$
\begin{aligned}
& y_{i} = \mu + \epsilon_{i}, \quad \epsilon_{i} \sim N(0, \sigma^{2}), \quad i = 1, ..., 15 \\
& y_{i} \sim N(\mu, \sigma^{2})
\end{aligned}
$$

Bayesian approach treats $\mu$ and $\sigma$ as random variable in a way of prior.<br><br>

Likelihood is a probability of data $P(y \mid \theta)$.<br>
Prior is a probability that characters uncertainty of parameters $P(\theta)$.<br>
A joint distribution $P(y, \theta) = P(y \mid \theta) P(\theta)$<br>
Posterior<br>
$$P(\theta \mid y) = \frac{P(\theta, y)}{P(y)} = \frac{P(\theta, y)}{\int P(\theta, y) d\theta} = \frac{P(y \mid \theta) P(\theta)}{\int P(y \mid \theta) P(\theta) d\theta}$$

<b>Q</b>: Whereas non-Bayesian approaches consider a probability model for the data only, the hallmark characteristic of Bayesian models is that they specify a joint probability distribution for both data and parameters. <b>How does the Bayesian paradigm leverage this additional assumption?</b><br>
A. This allows us to select the most accurate prior distribution.<br>
B. This allows us to use the laws of conditional probability to describe our updated information about parameters given the data.<br>
C. This allows us to make probabilistic assessments about how likely our particular data outcome is under any parameter setting.<br>
D. This allows us to make probabilistic assessments about hypothetical data outcomes given particular parameter values.<br>
<b>Answer</b>: B.<br><br>
</p>

#### 2.2.2 Model specification
<p align="justify">
$$y_{i} \mid \mu, \sigma^{2} \sim N(\mu, \sigma^{2}), \quad i = 1, 2, ..., 15$$

We regard $\mu$ and $\sigma^{2}$ are independent<br>
$$P(\mu, \sigma^{2}) = P(\mu) P(\sigma^{2})$$

If $\sigma^{2}$ is known, conjugate prior to $\mu$ is a normal distribution; if $\mu$ is known, the conjugate prior for $\sigma^{2}$ is a Inverse-Gamma distribution.<br>
$$
\begin{aligned}
& \mu \sim N(\mu_{0}, \sigma_{0}^{2}) \\
& \sigma^{2} \sim \text{Inverse Gamma}(\nu_{0}, \beta_{0})
\end{aligned}
$$

We can represent our model in a graph. $\mu$ and $\sigma^{2}$ are parameters to estimate. We use the parameters to generate the data. Here, data is observed (double circle).<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_2_2_1.png"/></center>
</p>
<p align="justify">
<b>Q</b>: <b>Which component of a Bayesian model would not appear in the model's hierarchical representation?</b><br>
A. Likelihood<br>
B. Prior<br>
C. Posterior<br>
<b>Answer</b>: C.<br><br>
</p>

#### 2.2.3 Posterior derivation
<p align="justify">
$$
\begin{aligned}
& y_{i} \mid \mu, \sigma^{2} \sim N(\mu, \sigma^{2}) \\
& \mu \mid \sigma^{2} \sim N(\mu_{0}, \frac{\sigma^{2}}{w_{0}}) \\
& \sigma^{2} \sim \text{Inverse-Gamma}(\nu_{0}, \beta_{0})
\end{aligned}
$$
We have 3 layers and we use a graph to represent it.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_2_3_1.png"/></center>
</p>
<p align="justify">
$$P(\theta \mid y) = \frac{P(y \mid \theta) P(\theta)}{\int P(y \mid \theta) P(\theta) d\theta} \propto P(y \mid \theta) P(\theta)$$
The joint distribution for whole variables<br>
$$
\begin{aligned}
P(y_{1}, ..., y_{n}, \mu, \sigma^{2}) & = P(y_{1}, ..., y_{n} \mid \mu, \sigma^{2}) P(\mu \mid \sigma^{2}) P(\sigma^{2}) \\
& = \prod_{i=1}^{n} [N(y_{i} \mid \mu, \sigma^{2})] \cdot N(\mu \mid \mu_{0}, \frac{\sigma^{2}}{w_{0}}) \cdot \text{IG}(\sigma^{2} \mid \nu_{0}, \beta_{0}) \\
& \propto P(\mu, \sigma^{2} \mid y_{1}, ..., y_{n})
\end{aligned}
$$

<b>Q</b>: When viewed as a function of some real number θ, the function
$$f(\theta) = \frac{\sigma}{\theta^{2} + 1} e^{-\sigma \theta}$$
<b>is proportional to which of the following?</b> Hint: To be proportional to the original function f(θ), the new function must be equal to f(θ) times a constant that does not involve θ, and this must be true for all possible values of θ.<br>
$$
\begin{aligned}
\text{A.} \quad & \frac{1}{\theta^{2} + 1} e^{-\theta}\\
\\
\text{B.}  \quad & \frac{1}{\theta^{2} + 1} e^{-\sigma \theta}\\
\\
\text{C.} \quad & \frac{\sigma}{\theta^{2}} e^{-\sigma \theta} \\
\\
\text{D.} \quad & \frac{1}{\theta^{2}} e^{-\theta}
\end{aligned}
$$
<b>Answer</b>: B.<br><br>
</p>

#### 2.2.4 Non-conjugate models
<p align="justify">
Suppose we have values that represent the percentage change in total personnel from last year to this year for n = 10 companies.<br>
$$
\begin{aligned}
& y_{i} \mid \mu \sim N(\mu, 1) \\
& \mu \sim t(0, 1,1)
\end{aligned}
$$
Posterior for $\mu$
$$
\begin{aligned}
P(\mu \mid y_{1}, ..., y_{n}) & \propto \prod_{i=1}^{n}[\frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2} (y_{i} - \mu)^{2}}] \frac{1}{\pi (1 + \mu^{2})} \\
& \propto e^{-\frac{1}{2} \sum_{i=1}^{n} (y_{i} - \mu)^{2}} \frac{1}{1 + \mu^{2}} \\
& \propto e^{-\frac{1}{2} (\sum_{i=1}^{n} y_{i}^{2} - 2\mu \sum_{i=1}^{n} y_{i} + n \mu^{2})}] \frac{1}{1 + \mu^{2}} \\
& \propto \frac{e^{n(\bar{y} \mu - \frac{\mu^{2}}{2})}}{1 + \mu^{2}}
\end{aligned}
$$
<b>Q</b>: <b>What major challenge do we face with both of the models introduced in this segment?</b><br>
A. The expression derived is only an approximation to the posterior.<br>
B. We have the posterior distribution up to a normalizing constant, but we are unable to integrate it to obtain important quantities, such as the posterior mean or probability intervals.<br>
C. The posterior distribution derived is not a proper probability distribution with a finite integral.<br>
D. We have the full posterior distribution, no methods exist for computing important quantities, such as the posterior mean or probability intervals.<br>
<b>Answer</b>: B.<br><br>
</p>

#### 2.2.5 Quiz
<p align="justify">
<b>1.</b><br>
Which of the following is one major difference between the frequentist and Bayesian approach to modeling data?<br>
A. Frequentist models are deterministic (don't use probability) while Bayesian models are stochastic (based on probability).<br>
B. The frequentist paradigm treats the data as fixed while the Bayesian paradigm considers data to be random.<br>
C. Frequentist models require a guess of parameter values to initialize models while Bayesian models require initial distributions for the parameters.<br>
D. Frequentists treat the unknown parameters as fixed (constant) while Bayesians treat unknown parameters as random variables.<br>
<b>Answer</b>: D.<br><br>

<b>2.</b><br>
Suppose we have a statistical model with unknown parameter θ, and we assume a normal prior $\theta \sim N(\mu_{0}, \sigma_{0}^{2})$, where $\mu_{0}$ is the prior mean and $\sigma_{0}^{2}$ is the prior variance. What does increasing $\sigma_{0}^{2}$ say about our prior beliefs about θ?<br>
A. Increasing the variance of the prior narrows the range of what we think θ might be, indicating less confidence in our prior mean guess $\mu_{0}$.<br>
B. Increasing the variance of the prior widens the range of what we think θ might be, indicating greater confidence in our prior mean guess $\mu_{0}$.<br>
C. Increasing the variance of the prior widens the range of what we think θ might be, indicating less confidence in our prior mean guess $\mu_{0}$.<br>
D. Increasing the variance of the prior narrows the range of what we think θ might be, indicating greater confidence in our prior mean guess $\mu_{0}$.<br>
<b>Answer</b>: C.<br><br>

<b>3.</b><br>
In the lesson, we presented Bayes' theorem for the case where parameters are continuous. What is the correct expression for the posterior distribution of θ if it is discrete (takes on only specific values)?
$$
\begin{aligned}
\text{A.} \quad & P(\theta) = \sum_{j} P(\theta \mid y_{j}) \cdot P(y_{i}) \\
\text{B.} \quad & P(\theta \mid y) = \frac{P(y \mid \theta) P(\theta)}{\int P(y \mid \theta) P(\theta) d\theta} \\
\text{C.} \quad & P(\theta_{j} \mid y) = \frac{P(y \mid \theta_{j}) P(\theta_{j})}{\sum_{j} P(y \mid \theta_{j}) P(\theta_{j})} \\
\text{D.} \quad & P(\theta) = \int P(\theta \mid y) \cdot P(y) dy
\end{aligned}
$$
<b>Answer</b>: C.<br><br>

<b>4. For Questions 4 and 5, refer to the following scenario.</b><br>
In the quiz for Lesson 1, we described Xie's model for predicting demand for bread at his bakery. During the lunch hour on a given day, the number of orders (the response variable) follows a Poisson distribution. All days have the same mean (expected number of orders). Xie is a Bayesian, so he selects a conjugate gamma prior for the mean with shape 3 and rate 1/15. He collects data on Monday through Friday for two weeks. Which of the following hierarchical models represents this scenario?<br>
$$
\begin{aligned}
\text{A.} & \quad
\begin{aligned}
&y_{i} \mid \lambda \sim \text{Pois}(\lambda), \quad \text{for } i = 1, ..., 10 \\
&\lambda \sim \text{Gamma}(3, \frac{1}{15})
\end{aligned}\\
\\
\text{B.} & \quad
\begin{aligned}
& y_{i} \mid \lambda \sim \text{Pois}(\lambda), \quad \text{for } i = 1, ..., 10 \\
& \lambda \mid \alpha \sim \text{Gamma}(\alpha, \frac{1}{15})\\
& \alpha \sim \text{Gamma}(3.0, 1.0)
\end{aligned} \\
\\
\text{C.} & \quad
\begin{aligned}
& y_{i} \mid \lambda \sim \text{Pois}(\lambda), \quad \text{for } i = 1, ..., 10 \\
& \mu \sim N(3, 15^{2})
\end{aligned} \\
\\
\text{D.} & \quad
\begin{aligned}
& y_{i} \mid \lambda \sim \text{Pois}(\lambda), \quad \text{for } i = 1, ..., 10 \\
& \lambda \mid \mu \sim \text{Gamma}(\mu, \frac{1}{15})\\
& \mu \sim \text{N}(3.0, 1.0^{2})
\end{aligned}
\end{aligned}
$$
<b>Answer</b>: A.<br><br>

<b>5.</b><br>
Which of the following graphical depictions represents the model from Xie's scenario?<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_2_5_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: A.<br><br>

<b>6.</b><br>
Graphical representations of models generally do not identify the distributions of the variables (nodes), but they do reveal the structure of dependence among the variables. Identify which of the following hierarchical models is depicted in the graphical representation below.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_2_5_2.png"/></center>
</p>
<p align="justify">
$$
\begin{aligned}
\text{A.} & \quad
\begin{aligned}
& x_{ij} \mid \alpha_{j}, \beta \sim \text{Gamma}(\alpha_{j}, \beta), \quad \text{for } i = 1, ..., n, \quad j = 1, ..., m \\
& \beta \sim \text{Exp}(b_{0}) \\
& \alpha_{j} \mid \phi \sim \text{Exp}(\phi), \quad j = 1, ..., m \\
& \phi \sim \text{Exp}(r_{0})
\end{aligned} \\
\\
\text{B.} & \quad
\begin{aligned}
& x_{ij} \mid \alpha, \beta \sim \text{Gamma}(\alpha, \beta), \quad \text{for } i = 1, ..., n, \quad j = 1, ..., m \\
& \beta \sim \text{Exp}(b_{0}) \\
& \alpha \sim \text{Exp}(a_{0})\\
& \phi \sim \text{Exp}(r_{0})
\end{aligned} \\
\\
\text{C.} & \quad
\begin{aligned}
& x_{ij} \mid \alpha_{j}, \beta \sim \text{Gamma}(\alpha_{j}, \beta), \quad \text{for } i = 1, ..., n, \quad j = 1, ..., m \\
& \beta \sim \text{Exp}(b_{0}) \\
& \alpha_{j} \sim \text{Exp}(a_{0}), \quad j = 1, ..., m\\
& \phi \sim \text{Exp}(r_{0})
\end{aligned} \\
\\
\text{D.} & \quad
\begin{aligned}
& x_{ij} \mid \alpha_{j}, \beta_{j} \sim \text{Gamma}(\alpha_{j}, \beta_{j}), \quad \text{for } i = 1, ..., n, \quad j = 1, ..., m \\
& \beta_{j} \mid \phi \sim \text{Exp}(\phi), \quad j = 1, ..., m \\
& \alpha_{i} \mid \phi \sim \text{Exp}(\phi), \quad i = 1, ..., n\\
& \phi \sim \text{Exp}(r_{0})
\end{aligned}
\end{aligned}
$$
<b>Answer</b>: A.<br><br>

<b>7.</b><br>
Consider the following model for a binary outcome y:<br>
$$
\begin{aligned}
& y_{i} \mid \theta_{i} \sim \text{Bern}(\theta_{i}), \quad i = 1, ..., 6 \\
& \theta_{i} \mid \alpha \sim \text{Beta}(\alpha, b_{0}), \quad i = 1, ..., 6 \\
& \alpha \sim \text{Exp}(r_{0})
\end{aligned}
$$
where $\theta_{i}$ is the probability of success on trial i. What is the expression for the joint distribution of all variables, written as $P(y_{1}, ..., y_{6}, \theta_{1}, ..., \theta_{6}, \alpha)$ and denoted by P(...)? You may ignore the indicator functions specifying the valid ranges of the variables (although the expressions are technically incorrect without them).<br>
Hint: The PMF for a Bernoulli random variable is<br>
$$f_{y}(y \mid \theta) = \theta^{y} (1 - \theta)^{1-y} \quad \text{for } y = 0 \text{ or } y = 1 \text{ and } 0 < \theta < 1$$
The PDF for a Beta random variable is<br>
$$f_{\theta}(\theta \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}, \quad 0 < \theta < 1, \alpha > 0, \beta > 0$$
The PDF for an exponential random variable is<br>
$$f_{\alpha}(\alpha \mid \lambda) = \lambda e^{-\lambda \alpha}, \quad \lambda, \alpha > 0$$
$$
\begin{aligned}
\text{A.} & \quad P(...) = \prod_{i=1}^{6} [\theta^{y_{i}} (1 - \theta)^{1 - y_{i}}] \cdot \frac{\Gamma(\alpha + b_{0})}{\Gamma(\alpha) \Gamma(b_{0})} \theta^{\alpha - 1} (1 - \theta)^{b_{0}-1} \cdot r_{0} e^{-r_{0} \alpha} \\
\text{B.} & \quad P(...) = \prod_{i=1}^{6} [\theta_{i}^{y_{i}} (1 - \theta_{i})^{1 - y_{i}} \cdot \frac{\Gamma(\alpha + b_{0})}{\Gamma(\alpha) \Gamma(b_{0})} \theta_{i}^{\alpha - 1} (1 - \theta_{i})^{b_{0}-1} \cdot r_{0} e^{-r_{0} \alpha}] \\
\text{C.} & \quad P(...) = \prod_{i=1}^{6} [\theta_{i}^{y_{i}} (1 - \theta_{i})^{1 - y_{i}} \cdot \frac{\Gamma(\alpha + b_{0})}{\Gamma(\alpha) \Gamma(b_{0})} \theta_{i}^{\alpha - 1} (1 - \theta_{i})^{b_{0}-1}] \\
\text{D.} & \quad P(...) = \prod_{i=1}^{6} [\theta_{i}^{y_{i}} (1 - \theta_{i})^{1 - y_{i}} \cdot \frac{\Gamma(\alpha + b_{0})}{\Gamma(\alpha) \Gamma(b_{0})} \theta_{i}^{\alpha - 1} (1 - \theta_{i})^{b_{0}-1}] \cdot r_{0} e^{-r_{0} \alpha}
\end{aligned}
$$
<b>Answer</b>: D.<br><br>

<b>8.</b><br>
In a Bayesian model, let y denote all the data and θ denote all the parameters. Which of the following statements about the relationship between the joint distribution of all variables p(y,θ)=p(⋯) and the posterior distribution p(θ∣y) is true?<br>
A. Neither is sufficient alone--they are both necessary to make inferences about θ.<br>
B. They are proportional to each other so that p(y,θ)=c⋅p(θ∣y) where c is a constant number that doesn't involve θ at all.<br>
C. The joint distribution p(y,θ) is equal to the posterior distribution times a function f(θ) which contains the modification (update) of the prior.<br>
D. They are actually equal to each other so that p(y,θ)=p(θ∣y).<br>
<b>Answer</b>: B.<br><br>
</p>

### 2.3 Monte Carlo Estimation
#### 2.3.1 Monte Carlo integration
<p align="justify">
$$
\begin{aligned}
& \theta \sim \Gamma(a = 2, b=\frac{1}{3}) \\
& E(\theta) = \int_{0}^{\infty} \theta p(\theta) d\theta = \int_{0}^{\infty} \theta \frac{b^{a}}{\Gamma(a)} \theta^{a- 1} e^{-b \theta} d\theta = \frac{a}{b}
\end{aligned}
$$

We use Monte Carlo estimation<br>
$$\theta_{i}^{*}, \quad i = 1, ..., m$$

Sample mean<br>
$$\bar{\theta^{*}} = \frac{1}{m} \sum_{i=1}^{m} \theta_{i}^{*}$$

Sample variance<br>
$$\text{Var}(\theta) = \int_{0}^{\infty} (\theta - E(\theta))^{2 p(\theta)} d\theta$$

So, for any function h($\theta$), we can estimate E(h($\theta$)) by sample mean.<br>
$$\int h(\theta) p(\theta) d\theta = E(h(\theta)) \approx \frac{1}{m} \sum_{i = 1}^{m} h(\theta_{i}^{m})$$

For example, h($\theta$) = $\mathbf{I}_{\theta < 5}(\theta)$, $\theta$ follows a Gamma distribution<br>
$$
\begin{aligned}
E(h(\theta)) & = \int_{0}^{5} \mathbf{I}_{\theta < 5} p(\theta) d\theta \\
& = \int_{0}^{5} 1 \cdot p(\theta) d\theta + \int_{5}^{\infty} 0 \cdot p(\theta) d\theta \\
& = P[0 < \theta < 5] \\
& \approx \frac{1}{m} \sum_{i=1}^{m} \mathbf{I}_{\theta^{*} < 5}(\theta_{i}^{*})
\end{aligned}
$$

<b>Q</b>: Forecasters often use simulations (usually based on a probability model) to approximate the probability of something they are trying to predict (for example, see https://fivethirtyeight.com/). <b>How do they use the simulations to obtain the forecast probability?</b><br>
A. They calculate the probability directly by integrating the probabilistic model. They then run one simulation, inputting the calculated probability. If the event occurs in the simulation, they forecast that it will occur.<br>
B. They calculate the probability directly within each simulation by integrating the probabilistic model. They then average these probabilities across many simulations.<br>
C. They simulate the system under study many times and count the fraction of times the event of interest occurs.<br>
D. They simulate the system under study once. If the event of interest occurs in that simulation, they forecast that it will occur.<br>
<b>Answer</b>: C.<br><br>
</p>

#### 2.3.2 Monte Carlo error and marginalization
<p align="justify">
Sample mean approximately follows a normal distribution<br>
$$\bar{\theta}^{*} \hat{\sim} N(E(\theta), \frac{\text{Var}(\theta)}{m})$$

To estimate the theoretical variance<br>
$$
\begin{aligned}
& \hat{\text{Var}(\theta)} = \frac{1}{m} \sum_{i=1}^{m} (\theta_{i}^{*} - \bar{\theta}^{*})^{2} \\
& \sqrt{\frac{\hat{\text{Var}(\theta)}}{m}} =  \text{ standard error}
\end{aligned}
$$

This applies also to hierarchical models. For example, y follows a binomial distribution and $\phi$ is from a Beta distribution.<br>
$$
\begin{aligned}
& y \mid \phi \sim \text{Bin}(10, \phi) \\
& \phi \sim \text{Beta} (2, 2) \\
& P(y, \phi) = P(\theta)P(y \mid \phi)
\end{aligned}
$$

To simulate<br>
1) draw $\theta_{i}^{*}$ from Beta(2, 2)<br>
2) given $\theta_{i}^{*}$, draw $y_{i}^{*}$ from Binomial(10, $\phi$)<br><br>

in this way, we can easily compute the marginal distribution from the joint distribution.<br><br>

<b>Q</b>: <b>What is the easiest way to increase accuracy of a Monte Carlo estimate?</b><br>
A. Increase the number of samples simulated.<br>
B. Change the random number generator seed.<br>
C. Discard samples that appear to be outliers.<br>
D. If sampling multiple variables, keep only the samples for the variable of interest.<br>
<b>Answer</b>: A.<br><br>
</p>

#### 2.3.3 Quiz
<p align="justify">
<b>1.</b><br>
If a random variable X follows a standard uniform distribution X∼Unif(0,1)), then the PDF of X is p(x)=1 for 0≤x≤1. We can use Monte Carlo simulation of X to approximate the following integral<br>
$$\int_{0}^{1} x^{2} dx = \int_{0}^{1} x^{2} \cdot 1 dx = \int_{0}^{1} x^{2} \cdot p(x) dx = E(x^{2})$$
If we simulate 1000 independent samples from the standard uniform distribution and call them $x_{i}^{*}$ for i = 1, ..., 1000, which of the following calculations will approximate the integral above?<br>
$$
\begin{aligned}
& \text{A.} \quad \frac{1}{1000} \sum_{i=1}^{1000} (x_{i}^{*})^{2} \\
\\
& \text{B.} \quad \frac{1}{1000} \sum_{i=1}^{1000} (x_{i}^{*} - \bar{x}^{*})^{2} \text{ where } \bar{x}^{*} \text{ is the calculated average of the } \bar{x}_{i}^{*} samples\\
\\
& \text{C.} \quad (\frac{1}{1000} \sum_{i=1}^{1000} x_{i}^{*})^{2} \\
\\
& \text{D.} \quad \frac{1}{1000} \sum_{i=1}^{1000} x_{i}^{*}
\end{aligned}
$$
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
Suppose we simulate 1000 samples from a Unif(0, $\pi$) distribution (which has PDF p(x) = $\frac{1}{\pi}$ for 0 $\leq$ x $\leq$ $\pi$ and call the samples $x_{i}^{*}$ for i = 1, ..., 1000. If we use these samples to calculate $\frac{1}{1000} \sum_{i=1}^{1000} \sin(x_{i}^{*})$. what integral are we approximating?<br>
$$
\begin{aligned}
& \text{A.} \quad \int_{0}^{1} \sin(x) dx \\
\\
& \text{B.} \quad \int_{0}^{1} \frac{\sin(x)}{x} dx  \\
\\
& \text{C.} \quad \int_{-\infty}^{\infty} \sin(x) dx \\
\\
& \text{D.} \quad \int_{0}^{\pi} \frac{\sin(x)}{\pi} dx
\end{aligned}
$$
<b>Answer</b>: D.<br><br>

<b>3.</b><br>
Suppose random variables X and Y have a joint probability distribution p(X,Y). Suppose we simulate 1000 samples from this distribution, which gives us 1000 $(x_{i}^{*}, y_{i}^{*})$ pairs. If we count how many of these pairs satisfy the condition $x_{i}^{*} < y_{i}^{*}$ and divide the result by 1000, what quantity are we approximating via Monte Carlo simulation?<br>
A. E(XY)<br>
B. P(X < Y)<br>
C. P(X < E(Y))<br>
D. P(E(X) < E(Y))<br>
<b>Answer</b>: B.<br><br>

<b>4.</b><br>
If we simulate 100 samples from a Gamma(2,1) distribution, what is the approximate distribution of the sample average <br>
$$\bar{x^{*}} = \frac{1}{100} \sum_{i=1}^{100} x_{i}^{*}$$
Hint: the mean and variance of a Gamma(a,b) random variable are a/b and $\frac{a}{b^{2}}$ respectively.<br>
A. Gamma(2, 1)<br>
B. N(2, 0.02)<br>
C. Gamma(2, 0.01<br>
D. N(2, 2)<br>
<b>Answer</b>: B.<br><br>

<b>5. For Questions 5 and 6, consider the following scenario:</b><br>
Laura keeps record of her loan applications and performs a Bayesian analysis of her success rate θ. Her analysis yields a Beta(5,3) posterior distribution for θ. The posterior mean for θ is equal to $\frac{5}{5+3}$ = 0.625. However, Laura likes to think in terms of the odds of succeeding, defined as $\frac{\theta}{1 - \theta}$, the probability of success divided by the probability of failure. Use R to simulate a large number of samples (more than 10,000) from the posterior distribution for θ and use these samples to approximate the posterior mean for Laura's odds of success E($\frac{\theta}{1-\theta}$)<br>
<b>Answer</b>: 2.498532.<br>
</p>
{% highlight R %}
theta = rbeta(10000000, 5, 3)
mean(theta/(1-theta))
{% endhighlight %}
<p align="justify">
<br>
<b>6.</b><br>
Laura also wants to know the posterior probability that her odds of success on loan applications is greater than 1.0 (in other words, better than 50:50 odds). Use your Monte Carlo sample from the distribution of θ to approximate the probability that $\frac{\theta}{1 - \theta}$ is greater than 1.0. Report your answer to at least two decimal places.<br>
<b>Answer</b>: 0.773.<br>
</p>
{% highlight R %}
theta = rbeta(9999, 5, 3)
alpha = theta / (1 - theta)
mean(alpha)
mean(alpha > 1.0)
{% endhighlight %}
<p align="justify">
<br>
<b>7.</b><br>
Use a (large) Monte Carlo sample to approximate the 0.3 quantile of the standard normal distribution (N(0,1)), the number such that the probability of being less than it is 0.3. Use the quantile function in R. You can of course check your answer using the qnorm function. Report your answer to at least two decimal places.<br>
<b>Answer</b>: -0.529.<br>
</p>
{% highlight R %}
quantile( rnorm(9999, 0.0, 1.0), 0.4 )
qnorm(0.3, 0.0, 1.0)
{% endhighlight %}
<p align="justify">
<br>
<b>8.</b><br>
To measure how accurate our Monte Carlo approximations are, we can use the central limit theorem. If the number of samples drawn m is large, then the Monte Carlo sample mean $\bar{\theta}^{*}$ used to estimate E(θ) approximately follows a normal distribution with mean E(θ) and variance Var(θ)/m. If we substitute the sample variance for Var(θ), we can get a rough estimate of our Monte Carlo standard error (or standard deviation). Suppose we have 100 samples from our posterior distribution for $\theta$, called $\theta_{i}^{*}$, and that the sample variance of these draws is 5.2. A rough estimate of our Monte Carlo standard error would then be $\sqrt{\frac{5.2}{100}} \approx$ 0.228. So our estimate $\bar{\theta}^{*}$is probably within about 0.456 (two standard errors) of the true E(θ). What does the standard error of our Monte Carlo estimate become if we increase our sample size to 5,000? Assume that the sample variance of the draws is still 5.2. Report your answer to at least three decimal places.<br>
<b>Answer</b>: 0.0325.<br>
$\sqrt{\frac{5.2}{500}}$<br><br>
</p>

#### 2.3.4 Reading: Markov chains
<p align="justify">
<b>Definition</b><br>
f we have a sequence of random variables $X_{1}$, $X_{2}$, .., $X_{n}$ where the indices 1, 2, ..., n represent successive points in time, we can use the chain rule of probability to calculate the probability of the entire sequence:<br>
$$P(X_{1}, X_{2}, ..., X_{n}) = P(X_{1}) \cdot P(X_{2} \mid X_{1}) \cdot P(X_{3} \mid X_{2}, X_{1}) \cdot ..., \cdot P(X_{n} \mid X_{n-1}, X_{n-2}, ..., X_{2}, X_{1})$$

Markov chains simplify this expression by using the Markov assumption. The assumption is that given the entire past history, the probability distribution for the random variable at the next time step only depends on the current variable. Mathematically, the assumption is written like this:<br>
$$P(X_{t+1} \mid X_{t}, X_{t-1}, ..., X_{2, X_{1}}) = P(X_{t+1} \mid X_{t}), \quad t = 2, .., n$$

Under this assumption, we can write the first expression as<br>
$$P(X_{1}, X_{2}, ..., X_{n}) = P(X_{1}) \cdot P(X_{2} \mid X_{1}) \cdot P(X_{3} \mid X_{2}) \cdot ... \cdot O(X_{n} \mid X_{n-1})$$

which is much simpler than the original. It consists of an initial distribution for the first variable, P($X_{1}$), and n−1 transition probabilities. We usually make one more assumption: that the transition probabilities do not change with time. Hence, the transition from time t to time t+1 depends only on the value of $X_{t}$.<br><br>

<b>Examples of Markov chains</b><br>
<b>Discrete Markov chain</b><br>
Suppose you have a secret number (make it an integer) between 1 and 5. We will call it your initial number at step 1. Now for each time step, your secret number will change according to the following rules:<br>
1. Flip a coin.<br>
2. If the coin turns up heads, then increase your secret number by one (5 increases to 1); If the coin turns up tails, then decrease your secret number by one (1 decreases to 5).<br>
3. Repeat n times, and record the evolving history of your secret number.<br><br>

Before the experiment, we can think of the sequence of secret numbers as a sequence of random variables, each taking on a value in {1,2,3,4,5}. Assume that the coin is fair, so that with each flip, the probability of heads and tails are both 0.5.<br><br>

Does this game qualify as a true Markov chain? Suppose your secret number is currently 4 and that the history of your secret numbers is (2,1,2,3). What is the probability that on the next step, your secret number will be 5? What about the other four possibilities? Because of the rules of this game, the probability of the next transition will depend only on the fact that your current number is 4. The numbers further back in your history are irrelevant, so this is a Markov chain.<br><br>

This is an example of a discrete Markov chain, where the possible values of the random variables come from a discrete set. Those possible values (secret numbers in this example) are called states of the chain. The states are usually numbers, as in this example, but they can represent anything. In one common example, the states describe the weather on a particular day, which could be labeled as 1-fair, 2-poor.<br><br>

<b>Random walk (continuous)</b><br>
Now let’s look at a continuous example of a Markov chain. Say $X_{t}$ = 0 and we have the following transition model: p($X_{t+1}$ | $X_{t}$ = $x_{t}$) = N($x_{t}$, 1). That is, the probability distribution for the next state is Normal with variance 1 and mean equal to the current state. This is often referred to as a “random walk.” Clearly, it is a Markov chain because the transition to the next state $X_{t+1}$ only depends on the current state $X_{t}$. This example is straightforward to code in R:<br>
</p>
{% highlight R %}
theta = rbeta(9999, 5, 3)
alpha = theta / (1 - theta)
mean(alpha)
mean(alpha > 1.0)
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_3_4_1.png"/></center>
</p>
<p align="justify">
<b>Transition matrix</b><br>
Let’s return to our example of the discrete Markov chain. If we assume that transition probabilities do not change with time, then there are a total of 25 ($5^{2}$ potential transition probabilities. Potential transition probabilities would be from State 1 to State 2, State 1 to State 3, and so forth. These transition probabilities can be arranged into a matrix Q:<br>
$$
Q =
\begin{bmatrix}
0 & 0.5 & 0 & 0 & 0.5 \\
0.5 & 0 & 0.5 & 0 & 0 \\
0 & 0.5 & 0 & 0.5 & 0 \\
0 & 0 & 0.5 & 0 & 0.5 \\
0.5 & 0 & 0 & 0.5 & 0
\end{bmatrix}
$$

where the transitions from State 1 are in the first row, the transitions from State 2 are in the second row, etc. For example, the probability $P(X_{t+1} = 5 \mid X_{t} = 4)$ can be found in the fourth row, fifth column.<br><br>

The transition matrix is especially useful if we want to find the probabilities associated with multiple steps of the chain. For example, we might want to know $P(X_{t+2} = 3 \mid X_{t} = 1)$, the probability of your secret number being 3 two steps from now, given that your number is currently 1. We can calculate this as<br>
$$\sum_{k=1}^{5} P(X_{t+2} = 3 \mid X_{t+1} = k) \cdot P(X_{t+1} = k \mid X_{t} = 1)$$

which conveniently is found in the first row and third column of $Q^{2}$. We can perform this matrix multiplication easily in R:<br>
</p>
{% highlight R %}
Q = matrix(c(0.0, 0.5, 0.0, 0.0, 0.5,
             0.5, 0.0, 0.5, 0.0, 0.0,
             0.0, 0.5, 0.0, 0.5, 0.0,
             0.0, 0.0, 0.5, 0.0, 0.5,
             0.5, 0.0, 0.0, 0.5, 0.0), 
           nrow=5, byrow=TRUE)

Q %*% Q # Matrix multiplication in R. This is Q^2.

(Q %*% Q)[1,3]
{% endhighlight %}
<p align="justify">
Therefore, if your secret number is currently 1, the probability that the number will be 3 two steps from now is .25.<br><br>

<b>Stationary distribution</b><br>
Suppose we want to know the probability distribution of the your secret number in the distant future, say p($X_{t+h}$ | $X_{t}$) where h is a large number. Let’s calculate this for a few different values of h.<br>
</p>
{% highlight R %}
Q5 = Q %*% Q %*% Q %*% Q %*% Q # h=5 steps in the future
round(Q5, 3)

# h=10 steps in the future
Q10 = Q %*% Q %*% Q %*% Q %*% Q %*% Q %*% Q %*% Q %*% Q %*% Q
round(Q10, 3)

Q30 = Q
for (i in 2:30) {
  Q30 = Q30 %*% Q
}
round(Q30, 3) # h=30 steps in the future
{% endhighlight %}
<p align="justify">
Notice that as the future horizon gets more distant, the transition distributions appear to converge. The state you are currently in becomes less important in determining the more distant future. If we let h get really large, and take it to the limit, all the rows of the long-range transition matrix will become equal to (.2,.2,.2,.2,.2). That is, if you run the Markov chain for a very long time, the probability that you will end up in any particular state is 1/5=.2 for each of the five states. These long-range probabilities are equal to what is called the stationary distribution of the Markov chain.<br><br>

The stationary distribution of a chain is the initial state distribution for which performing a transition will not change the probability of ending up in any given state. That is,<br>
</p>
{% highlight R %}
c(0.2, 0.2, 0.2, 0.2, 0.2) %*% Q
{% endhighlight %}
<p align="justify">
One consequence of this property is that once a chain reaches its stationary distribution, the stationary distribution will remain the distribution of the states thereafter.<br><br>

We can also demonstrate the stationary distribution by simulating a long chain from this example.<br>
</p>
{% highlight R %}
n = 5000
x = numeric(n)
x[1] = 1 # fix the state as 1 for time 1
for (i in 2:n) {
  # draw the next state from the intergers 1 to 5 with probabilities
  # from the transition matrix Q, based on the previous value of X.
  x[i] = sample.int(5, size=1, prob=Q[x[i-1],])
}
{% endhighlight %}
<p align="justify">
Now that we have simulated the chain, let’s look at the distribution of visits to the five states.<br>
</p>
{% highlight R %}
table(x) / n
{% endhighlight %}
<p align="justify">
The overall distribution of the visits to the states is approximately equal to the stationary distribution.<br><br>

As we have just seen, if you simulate a Markov chain for many iterations, the samples can be used as a Monte Carlo sample from the stationary distribution. This is exactly how we are going to use Markov chains for Bayesian inference. In order to simulate from a complicated posterior distribution, we will set up and run a Markov chain whose stationary distribution is the posterior distribution.<br><br>

It is important to note that the stationary distribution doesn’t always exist for any given Markov chain. The Markov chain must have certain properties, which we won’t discuss here. However, the Markov chain algorithms we’ll use in future lessons for Monte Carlo estimation are guaranteed to produce stationary distributions.<br><br>

<b>Continuous example</b><br>
The continuous random walk example we gave earlier does not have a stationary distribution. However, we can modify it so that it does have a stationary distribution.<br><br>

Let the transition distribution be<br>
$$P(X_{t+1} \mid X_{t} = x_{t}) = N(\phi x_{t}, 1), \quad -1 < \phi < 1$$

That is, the probability distribution for the next state is Normal with variance 1 and mean equal to ϕ times the current state. As long as ϕ is between −1 and 1, then the stationary distribution will exist for this model. Let’s simulate this chain for ϕ=−0.6.<br>
</p>
{% highlight R %}
set.seed(38)

n = 1500
x = numeric(n)
phi = -0.6

for (i in 2:n) {
  x[i] = rnorm(1, mean=phi*x[i-1], sd=1.0)
}

plot.ts(x)
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_3_4_2.png"/></center>
</p>
<p align="justify">
The theoretical stationary distribution for this chain is normal with mean 0 and variance 1/(1−$\phi^{2}$), which in our example approximately equals 1.562. Let’s look at a histogram of our chain and compare that with the theoretical stationary distribution.<br>
</p>
{% highlight R %}
hist(x, freq=FALSE)
curve(dnorm(x, mean=0.0, sd=sqrt(1.0/(1.0-phi^2))), col="red", add=TRUE)
legend("topright", legend="theoretical stationary\ndistribution",
       col="red", lty=1, bty="n")
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_3_4_3.png"/></center>
</p>
<p align="justify">
It appears that the chain has reached the stationary distribution. Therefore, we could treat this simulation from the chain like a Monte Carlo sample from the stationary distribution, a normal with mean 0 and variance 1.562.<br><br>

Because most posterior distributions we will look at are continuous, our Monte Carlo simulations with Markov chains will be similar to this example.<br><br>
</p>

#### 2.3.5 Reading: Code for Lesson 3
{% highlight R %}
# Initializes the random number generator so we can replicate these results.
# To get different random numbers, change the seed.
set.seed(32)
m = 100
a = 2.0
b = 1.0 / 3.0

theta = rgamma(n=m, shape=a, rate=b)
hist(theta, freq=FALSE)
curve(dgamma(x=x, shape=a, rate=b), col="blue", add=TRUE)

sum(theta) / m # sample mean
mean(theta) # sample mean
a / b # true expected value

m = 1e4
theta = rgamma(n=m, shape=a, rate=b)
mean(theta)

var(theta) # sample variance
a / b^2 # true variance of Gamma(a,b)

ind = theta < 5.0 # set of indicators, TRUE if theta_i < 5
mean(ind) # automatically converts FALSE/TRUE to 0/1

pgamma(q=5.0, shape=a, rate=b) # true value of Pr( theta < 5 )

# What is the 0.9 quantile (90th percentile) of θ?
# We can use the quantile function which will order
# the samples for us and find the appropriate sample quantile.
quantile(x=theta, probs=0.9)

qgamma(p=0.9, shape=a, rate=b) # true value of 0.9 quantile

se = sd(theta) / sqrt(m)
# we are reasonably confident that the Monte Carlo estimate
# is no more than this far from the truth
2.0 * se

ind = theta < 5.0
se = sd(ind) / sqrt(m)
# we are reasonably confident that the Monte Carlo estimate is
# no more than this far from the truth
2.0 * se

# Marginalization
m = 10e4
y = numeric(m) # create the vectors we will fill in with simulations
phi = numeric(m)
for (i in 1:m) {
  phi[i] = rbeta(n=1, shape1=2.0, shape2=2.0)
  y[i] = rbinom(n=1, size=10, prob=phi[i])
}
# which is equivalent to the following 'vectorized' code
phi = rbeta(n=m, shape1=2.0, shape2=2.0)
y = rbinom(n=m, size=10, prob=phi)
mean(y)
plot(prop.table(table(y)), ylab="P(y)", main="Marginal distribution of y")
{% endhighlight %}
<p align="justify">
<br>
</p>

### 2.4 Metropolis-Hastings
#### 2.4.1 Algorithm
<p align="justify">
Let’s say we wish to produce samples from a target distribution $P(\theta) \propto g(\theta)$, where we don’t know the normalizing constant (since $\int g(\theta) d\theta$ is hard or impossible to compute), so we only have $g(\theta)$ to work with.<br><br>

Algorithms<br>
1) select initial value $\theta_{0}$<br>
2) for i = 1, ..., m repeat<br>
&emsp;a) draw candidate $\theta^{*} \sim q(\theta^{*} \mid \theta_{i-1})$<br>
&emsp;b) calculate $\alpha$<br>
$$\alpha = \frac{g(\theta^{*}) / q(\theta^{*} \mid \theta_{i-1})}{g(\theta_{i-1}) / q(\theta_{i-1} \mid \theta^{*})} = \frac{g(\theta^{*}) q(\theta_{i-1} \mid \theta^{*})}{g(\theta_{i-1}) q(\theta^{*} \mid \theta_{i-1})}$$
&emsp;c) check $\alpha$<br>
&emsp;&emsp;$\alpha \geq$ 1, accept $\theta^{*}$ and set $\theta_{i} \leftarrow \theta^{*}$<br>
&emsp;&emsp;0 < $\alpha$ < 1<br>
&emsp;&emsp;&emsp;accept $\theta^{*}$ and set $\theta_{i} \leftarrow \theta^{*}$  with probability $\alpha$<br>
&emsp;&emsp;&emsp;reject $\theta^{*}$ and set $\theta_{i} \leftarrow \theta_{i-1}$ with probability $1 - \alpha$<br><br>

<b>Q</b>: <b>What is the advantage of using a symmetric proposal distribution $q(\theta^{*} \mid \theta_{i-1})$ in a random walk Metropolis-Hastings sampler?</b><br>
A. Symmetry in a random walk causes the acceptance ratio to be greater than 1. Hence, the candidate is always accepted and we avoid an extra calculation.<br>
B. Symmetry in a random walk yields $q(\theta^{*} \mid \theta_{i-1}) = q(\theta_{i-1} \mid \theta^{*})$ causing both expressions to drop out of the acceptance calculation which then becomes $g(\theta^{*}) / g(\theta_{i-1})$<br>
C. Symmetry in a random walk yields $g(\theta^{*}) = g(\theta_{i-1})$, causing both expressions to drop out of the acceptance calculation which then becomes $q(\theta_{i-1} \mid \theta^{*}) / q(\theta^{*} \mid \theta_{i-1})$<br>
D. Symmetry in a random walk provides a default choice that will not need to be tweaked for performance.<br>
<b>Answer</b>: B.<br><br>

<b>Proposal distribution</b><br>
One careful choice we must make is the candidate generating distribution $q(\theta^{*} \mid \theta_{i-1})$. It may or may not depend on the previous iteration’s value of θ. One example where it doesn’t depend on the previous value would be if $q(\theta^{*})$ is always the same distribution. If we use this option, q(θ) should be as similar as possible to p(θ). Another popular option, one that does depend on the previous iteration, is Random-Walk Metropolis-Hastings. Here, the proposal distribution is centered on $\theta_{i-1}$. For instance, it might be a normal distribution with mean $\theta_{i-1}$. Because the normal distribution is symmetric, this example comes with another advantage $q(\theta^{*} \mid \theta_{i-1})$ = $q(\theta_{i-1} \mid \theta^{*})$, causing it to cancel out when we calculate α. Thus, in Random-Walk Metropolis-Hastings where the candidate is drawn from a normal with mean $\theta_{i-1}$ and constant variance, the acceptance ratio is $\alpha$ = $g(\theta^{*})$ / $g(\theta_{i-1})$.<br><br>

<b>Acceptance rate</b><br>
Clearly, not all candidate draws are accepted, so our Markov chain sometimes “stays” where it is, possibly for many iterations. How often you want the chain to accept candidates depends on the type of algorithm you use. If you approximate P($\theta$) with $q(\theta^{*})$ and always draw candidates from that, accepting candidates often is good; it means $q(\theta^{*})$ is approximating P($\theta$) well. However, you still may want q to have a larger variance than p and see some rejection of candidates as an assurance that q is covering the space well. As we will see in coming examples, a high acceptance rate for the Random-Walk Metropolis-Hastings sampler is not a good thing. If the random walk is taking too small of steps, it will accept often, but will take a very long time to fully explore the posterior. If the random walk is taking too large of steps, many of its proposals will have low probability and the acceptance rate will be low, wasting many draws. Ideally, a random walk sampler should accept somewhere between 23% and 50% of the candidates proposed.<br><br>
</p>

#### 2.4.2 Demonstration
<p align="justify">
Take the coin as an example<br>
$$\theta = \{\text{fair}, \text{loaded}\} $$
Prior<br>
$$P(\text{loaded}) = 0.6$$
Posterior<br>
$$
\begin{aligned}
f(\theta \mid x) & = \frac{f(x \mid \theta) p(\theta)}{\sum_{\theta} f(x \mid \theta) p(\theta)} \\
& = \frac{\binom{5}{x} [(0.5)^{5} (0.4) I_{\theta = \text{fair}}+ (0.7)^{x}(0.3)^{5-x}(0.6) I_{\theta = \text{loaded}}]}{\binom{5}{x} [(0.5)^{5} (0.4) + (0.7)^{x} (0.3)^{5-x} (0.6)]}
\end{aligned}
$$
If we get 2 heads and 3 tails<br>
$$
\begin{aligned}
f(\theta \mid X =2) & = \frac{0.0125 I_{\theta = \text{fair}} + 0.0079 I_{\theta = \text{loaded}}}{0.0125 + 0.0079} \\
& = 0.612 I_{\theta = \text{fair}} + 0.388 I_{\theta = \text{loaded}}
\end{aligned}
$$
Then based on the data, the probability that this coin is loaded is<br>
$$P(\theta = \text{loaded} \mid X = 2) = 0.388$$
Now, we use Metropolis-Hastings to approximate the posterior. Here are two states: fair and loaded.<br>
1) start at an arbitrary location / state: $\theta_{0}$ = fair or $\theta_{0}$ = loaded<br>
2) For i = 1, ..., m iterate<br>
&emsp;a) propose a candidate $\theta^{*}$ to be the other state as $\theta_{i-1}$ (e.g. cuurent state is fair, propose state $\theta_{*}$ is loaded)<br>
&emsp;b) calculate $\alpha$ with g($\theta$) = likelihood $\times$ prior and q($\theta$ | $\theta'$) = 1<br>
$$
\begin{aligned}
\alpha & = \frac{g(\theta^{*}) / q(\theta^{*} \mid \theta_{i-1})}{g(\theta_{i-1}) / q(\theta_{i-1} \mid \theta^{*})} \\
& = \frac{g(\theta^{*}) q(\theta_{i-1} \mid \theta^{*})}{g(\theta_{i-1}) q(\theta^{*} \mid \theta_{i-1})} \\
& = \frac{f(X = 2 \mid \theta^{*}) f(\theta^{*}) / 1}{f(X = 2 \mid \theta_{i-1}) f(\theta_{i-1}) / 1} \\
& =
\begin{cases}
\frac{0.00794}{0.0125} = 0.635, \quad \theta^{*} = \text{loaded} \\
\frac{0.0125}{0.00794} = 1.574, \quad  \theta^{*} = \text{fair}
\end{cases}
\end{aligned}
$$
&emsp;c) accpet or reject<br>
&emsp;&emsp;If $\theta^{*}$ is fair, $\alpha$ > 1, accept $\theta^{*}$ and set $\theta_{i}$ fair.<br>
&emsp;&emsp;If $\theta^{*}$ is loaded, 0 < $\alpha$ < 1,<br>
&emsp;&emsp;&emsp;accept $\theta^{*}$ with probability of 0.635. set $\theta_{i}$ loaded<br>
&emsp;&emsp;&emsp;reject $\theta^{*}$ with probability of 0.365. set $\theta_{i}$ fair<br><br>

We can draw a schema for the Markov chain<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_2_1.png"/></center>
</p>
<p align="justify">
Stationary distribution $\pi$<br>
$$\pi \cdot P = \pi, \quad P \text{ is a transition matrix}$$
$$
P =
\begin{bmatrix}
0.365 & 0.635 \\
1 & 0
\end{bmatrix}
\quad
\pi =
\begin{bmatrix}
0.612 & 0.388 \\
\end{bmatrix}
$$
We get a same value of posterior for $\theta$ is faire and $\theta$ is loaded.<br><br>

<b>Q</b>: If we were to simulate this Markov chain for many iterations, approximately <b>what fraction of the time would the chain be in θ=fair?</b><br>
A. 0.635<br>
B. 0.388<br>
C. 0.365<br>
D. 0.612<br>
<b>Answer</b>: D.<br><br>
</p>

#### 2.4.3 Random walk example
<p align="justify">
<b>Random walk with normal likelihood, t prior</b><br>
Recall the model from the last segment of Lesson 2 where the data are the percent change in total personnel from last year to this year for n=10 companies. We used a normal likelihood with known variance and t distribution for the prior on the unknown mean. Suppose the values are y=(1.2,1.4,−0.5,0.3,0.9,2.3,1.0,0.1,1.3,1.9). Because this model is not conjugate, the posterior distribution is not in a standard form that we can easily sample. To obtain posterior samples, we will set up a Markov chain whose stationary distribution is this posterior distribution. Recall that the posterior distribution is<br>
$$P(\mu \mid y_{1}, ..., y_{n}) \propto \frac{e^{n(\bar{y} \mu - \frac{\mu^{2}}{2})}}{1 + \mu^{2}}$$
The posterior distribution on the left is our target distribution and the expression on the right is our g(μ).<br><br>

The first thing we can do in R is write a function to evaluate g(μ). Because posterior distributions include likelihoods (the product of many numbers that are potentially small), g(μ) might evaluate to such a small number that to the computer, it effectively zero. This will cause a problem when we evaluate the acceptance ratio α. To avoid this problem, we can work on the log scale, which will be more numerically stable. Thus, we will write a function to evaluate<br>
$$\log(g(\mu)) = n(\bar{y} \mu - \frac{\mu^{2}}{2}) - \log(1 + \mu^{2})$$
This function will require three arguments, μ, $\bar{y}$, and n.<br>
</p>
{% highlight R %}
lg = function(mu, n, ybar) {
  mu2 = mu^2
  n * (ybar * mu - mu2 / 2.0) - log(1 + mu2)
}
{% endhighlight %}
<p align="justify">
Next, let’s write a function to execute the Random-Walk Metropolis-Hastings sampler with normal proposals.<br>
</p>
{% highlight R %}
mh = function(n, ybar, n_iter, mu_init, cand_sd) {
  ## Random-Walk Metropolis-Hastings algorithm
  
  ## step 1, initialize
  mu_out = numeric(n_iter)
  accpt = 0
  mu_now = mu_init
  lg_now = lg(mu=mu_now, n=n, ybar=ybar)
  
  ## step 2, iterate
  for (i in 1:n_iter) {
    ## step 2a
    mu_cand = rnorm(n=1, mean=mu_now, sd=cand_sd) # draw a candidate
    
    ## step 2b
    # evaluate log of g with the candidate
    lg_cand = lg(mu=mu_cand, n=n, ybar=ybar)
    lalpha = lg_cand - lg_now # log of acceptance ratio
    alpha = exp(lalpha)
    
    ## step 2c
    # draw a uniform variable which will be less than alpha with
    # probability min(1, alpha)
    u = runif(1)
    if (u < alpha) { # then accept the candidate
      mu_now = mu_cand
      accpt = accpt + 1 # to keep track of acceptance
      lg_now = lg_cand
    }
    
    ## collect results
    mu_out[i] = mu_now # save this iteration's value of mu
  }
  
  ## return a list of output
  list(mu=mu_out, accpt=accpt/n_iter)
}
{% endhighlight %}
<p align="justify">
Now, let’s set up the problem.<br>
</p>
{% highlight R %}
y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)
ybar = mean(y)
n = length(y)
hist(y, freq=FALSE, xlim=c(-1.0, 3.0)) # histogram of the data
curve(dt(x=x, df=1), lty=2, add=TRUE) # prior for mu
points(y, rep(0,n), pch=1) # individual data points
points(ybar, 0, pch=19) # sample mean
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_3_1.png"/></center>
</p>
<p align="justify">
Finally, we’re ready to run the sampler! Let’s use m=1000 iterations and proposal standard deviation (which controls the proposal step size) 3.0, and initial value at the prior median 0.<br>
</p>
{% highlight R %}
set.seed(43) # set the random seed for reproducibility
post = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=3.0)
str(post)

library("coda")
traceplot(as.mcmc(post$mu))
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_3_2.png"/></center>
</p>
<p align="justify">
This last plot is called a trace plot. It shows the history of the chain and provides basic feedback about whether the chain has reached its stationary distribution. It appears our proposal step size was too large (acceptance rate below 23%). Let’s try another.<br>
</p>
{% highlight R %}
post = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=0.05)
post$accpt

traceplot(as.mcmc(post$mu))
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_3_3.png"/></center>
</p>
<p align="justify">
Oops, the acceptance rate is too high (above 50%). Let’s try something in between.<br>
</p>
{% highlight R %}
post = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=0.9)
post$accpt

traceplot(as.mcmc(post$mu))
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_3_4.png"/></center>
</p>
<p align="justify">
Hey, that looks pretty good. Just for fun, let’s see what happens if we initialize the chain at some far-off value.<br>
</p>
{% highlight R %}
post = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=30.0, cand_sd=0.9)
post$accpt

traceplot(as.mcmc(post$mu))
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_3_5.png"/></center>
</p>
<p align="justify">
It took awhile to find the stationary distribution, but it looks like we succeeded! If we discard the first 100 or so values, it appears like the rest of the samples come from the stationary distribution, our posterior distribution! Let’s plot the posterior density against the prior to see how the data updated our belief about μ.<br>
</p>
{% highlight R %}
post$mu_keep = post$mu[-c(1:100)] # discard the first 200 samples
plot(density(post$mu_keep, adjust=2.0), main="", xlim=c(-1.0, 3.0),
     xlab=expression(mu)) # plot density estimate of the posterior
curve(dt(x=x, df=1), lty=2, add=TRUE) # prior for mu
points(ybar, 0, pch=19) # sample mean

# approximation to the true posterior in blue
curve(0.017*exp(lg(mu=x, n=n, ybar=ybar)),
      from=-1.0, to=3.0, add=TRUE, col="blue")
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_3_6.png"/></center>
</p>
<p align="justify">
These results are encouraging, but they are preliminary. We still need to investigate more formally whether our Markov chain has converged to the stationary distribution. We will explore this in a future lesson.<br><br>

Obtaining posterior samples using the Metropolis-Hastings algorithm can be time-consuming and require some fine-tuning, as we’ve just seen. The good news is that we can rely on software to do most of the work for us. In the next couple of videos, we’ll introduce a program that will make posterior sampling easy.<br><br>
</p>

#### 2.4.4 Quiz
<p align="justify">
<b>1.</b><br>
In which situation would we choose to use a Metropolis-Hastings (or any MCMC) sampler rather than straightforward Monte Carlo sampling?<br>
A. The target distribution follows a Markov chain.<br>
B. There is no easy way to simulate independent draws from the target distribution.<br>
C. The data (likelihood) come from a Markov chain.<br>
D. Monte Carlo estimation is easier than calculating the integral required to obtain the mean of the target distribution.<br>
<b>Answer</b>: B.<br><br>

<b>2.</b><br>
Which of the following candidate-generating distributions would be best for an independent Metropolis-Hastings algorithm to sample the target distribution whose PDF is shown below? Note: In independent Metropolis-Hastings, the candidate-generating distribution q does not depend on the previous iteration of the chain.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_4_1.png"/></center>
</p>
<p align="justify">
A. q = N(15, $3.1^{2}$)<br>
B. q = Gamma(3, 0.27)<br>
C. q = Unif(0.05, 25)<br>
D. q = Exp(0.1)<br>
<b>Answer</b>: .B<br>
This candidate-generating distribution approximates the target distribution well, and even has slightly larger variance.<br><br>

<b>3.</b><br>
If we employed an independent Metropolis-Hastings algorithm (in which the candidate-generating distribution q does not depend on the previous iteration of the chain), what would happen if we skipped the acceptance ratio step and always accepted candidate draws?<br>
A. Each draw could be considered as a sample from the target distribution.<br>
B. The resulting sample would be a Monte Carlo simulation from q instead of from the target distribution.<br>
C. The chain would explore the posterior distribution very slowly, requiring more samples.<br>
D. The sampler would become more efficient because we are no longer discarding draws.<br>
<b>Answer</b>: B.<br>
Accepting all candidates just means we are simulating from the candidate-generating distribution. The acceptance step in the algorithm acts as a correction, so that the samples reflect the target distribution more than the candidate-generating distribution.<br><br>

<b>4.</b><br>
If the target distribution p($\theta$) $\propto$ g($\theta$) is for a positive-valued random variable so that p(θ) contains the indicator function $\mathbf{I}_{\theta > 0}(\theta)$, what would happen if a random walk Metropolis sampler proposed the candidate $\theta^{*}$ = -0.3?<br>
A. The candidate would be rejected with probability 1 because g($\theta^{*}$) = 0 yielding an acceptance ratio $\alpha$ = 0.<br>
B. The candidate would be accepted with probability 0.3 because g($\theta^{*}$) = \left | \theta^{*} \right |, yielding an acceptance ratio $\alpha$ = 0.3.<br>
C. The candidate would be accepted with probability 1 ecause g($\theta^{*}$) = 0 yielding an acceptance ratio $\alpha$ = $\infty$.<br>
D. The candidate would be accepted with probability 1 ecause g($\theta^{*}$) = 0 yielding an acceptance ratio $\alpha$ = 1.<br>
<b>Answer</b>: A.<br><br>

<b>5.</b><br>
Suppose we use a random walk Metropolis sampler with normal proposals (centered on the current value of the chain) to sample from the target distribution whose PDF is shown below. The chain is currently at $\theta_{i}$ = 15.0. Which of the other points, if used as a candidate $\theta^{*}$ for the next step, would yield the largest acceptance ratio α?<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_4_2.png"/></center>
</p>
<p align="justify">
A. $\theta^{*}$ = 3.1<br>
B. $\theta^{*}$ = 9.8<br>
C. $\theta^{*}$ = 20.3<br>
D. $\theta^{*}$ = 26.1<br>
<b>Answer</b>: B.<br>
Because of normal proposal, $q(\theta^{*} \mid \theta_{i-1})$ = $q(\theta_{i-1} \mid \theta^{*})$<br><br>

<b>6.</b><br>
Suppose you are using a random walk Metropolis sampler with normal proposals. After sampling the chain for 1000 iterations, you notice that the acceptance rate for the candidate draws is only 0.02. Which corrective action is most likely to help you approach a better acceptance rate (between 0.23 and 0.50)?<br>
A. Increase the variance of the normal proposal distribution q.<br>
B. Decrease the variance of the normal proposal distribution q.<br>
C. Replace the normal proposal distribution with a uniform proposal distribution centered on the previous value and variance equal to that of the old normal proposal distribution.<br>
D. Fix the mean of the normal proposal distribution at the last accepted candidate's value. Use the new mean for all future proposals.<br>
<b>Answer</b>: B.<br>
A low acceptance rate in a random walk Metropolis sampler usually indicates that the candidate-generating distribution is too wide and is proposing draws too far away from most of the target mass.<br><br>

<b>7.</b><br>
Suppose we use a random walk Metropolis sampler to sample from the target distribution p($\theta$) $\propto$ g($\theta$) and propose candidates $\theta^{*}$ using Unif($\theta_{i-1}-\epsilon$, $\theta_{i+1} + \epsilon$) distribution where ϵ is some positive number and $\theta_{i-1}$ is the previous iteration's value of the chain. What is the correct expression for calculating the acceptance ratio α in this scenario? Hint: Notice that the Unif($\theta_{i-1}-\epsilon$, $\theta_{i+1} + \epsilon$) distribution is centered on the previous value and is symmetric (since the PDF is flat and extends the same distance ϵ on either side).<br>
$$
\begin{aligned}
& \text{A.} \quad \alpha = \frac{g(\theta^{*})}{g(\theta_{i-1})} \\
\\
& \text{B.} \quad \alpha = \frac{g(\theta_{i-1})}{g(\theta^{*})} \\
\\
& \text{C.} \quad \alpha = \frac{\text{Unif}(\theta^{*} \mid \theta_{i-1}-\epsilon, \theta_{i-1}+\epsilon)}{\text{Unif}(\theta_{i-1} \mid \theta^{*} - \epsilon, \theta^{*} + \epsilon)} \\
\\
& \text{D.} \quad \alpha = \frac{\text{Unif}(\theta_{i-1} \mid \theta^{*} - \epsilon, \theta^{*} + \epsilon)}{\text{Unif}(\theta^{*} \mid \theta_{i-1}-\epsilon, \theta_{i-1}+\epsilon)}
\end{aligned}
$$
<b>Answer</b>: A.<br>
Since the proposal distribution is centered on the previous value and is symmetric, evaluations of q drop from the calculation of α.<br><br>

<b>8.</b><br>
The following code completes one iteration of an algorithm to simulate a chain whose stationary distribution is p($\theta$) $\propto$ g($\theta$). Which algorithm is employed here?<br>
</p>
{% highlight R %}
# draw candidate
theta_cand = rnorm(n=1, mean=0.0, sd=10.0)

# evaluate log of g with the candidate
lg_cand = lg(theta=theta_cand)

# evaluate log of g at the current value
lg_now = lg(theta=theta_now)

# evaluate log of q at candidate
lq_cand = dnorm(theta_cand, mean=0.0, sd=10.0, log=TRUE)

# evaluate log of q at the current value
lq_now = dnorm(theta_now, mean=0.0, sd=10.0, log=TRUE)

# calculate the acceptance ratio
lalpha = lg_cand + lq_now - lg_now - lq_cand 
alpha = exp(lalpha)

# draw a uniform variable which will be less than alpha with
# probability min(1, alpha)
u = runif(1)

if (u < alpha) { # then accept the candidate
  theta_now = theta_cand
  accpt = accpt + 1 # to keep track of acceptance
}
{% endhighlight %}
<p align="justify">
A. Random walk Metropolis with uniform proposal<br>
B. Independent Metropolis-Hastings (q does not condition on the previous value of the chain) with uniform proposal<br>
C. Independent Metropolis-Hastings (q does not condition on the previous value of the chain) with normal proposal<br>
D. Random walk Metropolis with normal proposal<br>
<b>Answer</b>: C.<br>
Candidates are always drawn from the same N(0,$10$^{2}) distribution.<br><br>
</p>

#### 2.4.5 JAGS
<p align="justify">
JAGS is <a href="http://mcmc-jags.sourceforge.net">Just Another Gibbs Sampler.</a><br><br>

<b>Introduction to JAGS</b><br>
There are several software packages available that will handle the details of MCMC for us. See the supplementary material for a brief overview of options.<br><br>

The package we will use in this course is JAGS (Just Another Gibbs Sampler) by Martyn Plummer. The program is free, and runs on Mac OS, Windows, and Linux. Better yet, the program can be run using R with the rjags and R2jags packages.<br><br>

In JAGS, we can specify models and run MCMC samplers in just a few lines of code; JAGS does the rest for us, so we can focus more on the statistical modeling aspect and less on the implementation. It makes powerful Bayesian machinery available to us as we can fit a wide variety of statistical models with relative ease.<br><br>

<b>Installation and setup</b><br>
The starting place for JAGS users is mcmc-jags.sourceforge.net. At this site, you can find news about the features of the latest release of JAGS, links to program documentation, as well as instructions for installation.<br><br>

The documentation is particularly important. It is available under the files page link in the Manuals folder.<br><br>

Also under the files page, you will find the JAGS folder where you can download and install the latest version of JAGS. Select the version and operating system, and follow the instructions for download and installation.<br><br>

Once JAGS is installed, we can immediately run it from R using the rjags package. The next segment will show how this is done.<br><br>

<b>Modeling in JAGS</b><br>
There are four steps to implementing a model in JAGS through R:<br>
Specify the model.<br>
Set up the model.<br>
Run the MCMC sampler.<br>
Post processing.<br>
We will demonstrate these steps with our running example with the data are the percent change in total personnel from last year to this year for n=10 companies. We used a normal likelihood with known variance and t distribution for the prior on the unknown mean.<br><br>

<b>1. Specify the model</b><br>
In this step, we give JAGS the hierarchical structure of the model, assigning distributions to the data (the likelihood) and parameters (priors). The syntax for this step is very similar to R, but there are some key differences.<br>
</p>
{% highlight R %}
library("rjags")

mod_string = " model {
for (i in 1:n) {
y[i] ~ dnorm(mu, 1.0/sig2)
}
mu ~ dt(0.0, 1.0/1.0, 1.0) # location, inverse scale, degrees of freedom
sig2 = 1.0
} "
{% endhighlight %}
<p align="justify">
One of the primary differences between the syntax of JAGS and R is how the distributions are parameterized. Note that the normal distribution uses the mean and precision (instead of variance). When specifying distributions in JAGS, it is always a good idea to check the JAGS user manual in the chapter on Distributions.<br><br>

<b>2. Set up the model</b><br>
</p>
{% highlight R %}
set.seed(50)
y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)
n = length(y)

data_jags = list(y=y, n=n)
params = c("mu")

inits = function() {
  inits = list("mu"=0.0)
} # optional (and fixed)

mod = jags.model(textConnection(mod_string), data=data_jags, inits=inits)

## Compiling model graph
##    Resolving undeclared variables
##    Allocating nodes
## Graph information:
##    Observed stochastic nodes: 10
##    Unobserved stochastic nodes: 1
##    Total graph size: 28
## 
## Initializing model
{% endhighlight %}
<p align="justify">
There are multiple ways to specify initial values here. They can be explicitly set, as we did here, or they can be random, i.e., list("mu"=rnorm(1)). Also, we can omit the initial values, and JAGS will provide them.<br><br>

<b>3. Run the MCMC sampler</b><br>
</p>
{% highlight R %}
update(mod, 500) # burn-in

mod_sim = coda.samples(model=mod,
                       variable.names=params,
                       n.iter=1000)
{% endhighlight %}
<p align="justify">
<b>4. Post processing</b><br>
</p>
{% highlight R %}
summary(mod_sim)
## 
## Iterations = 1501:2500
## Thinning interval = 1 
## Number of chains = 1 
## Sample size per chain = 1000 
## 
## 1. Empirical mean and standard deviation for each variable,
##    plus standard error of the mean:
## 
##           Mean             SD       Naive SE Time-series SE 
##       0.895867       0.301648       0.009539       0.012256 
## 
## 2. Quantiles for each variable:
## 
##   2.5%    25%    50%    75%  97.5% 
## 0.3153 0.6854 0.8888 1.0937 1.4977

library("coda")
plot(mod_sim)
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_4_5_1.png"/></center>
<br>
</p>

### 2.5 Gibbs Sampling
#### 2.5.1 Multiple parameter sampling and full conditional distributions
<p align="justify">
We have two parameters $\theta$ and $\phi$. $g(\theta, \phi)$ is a normalized format
$$P(\theta, \phi \mid y) \propto g(\theta, \phi)$$
For each parameter, we have a full conditional probability
$$
\begin{aligned}
& P(\theta \mid \phi, y) \propto P(\theta, \phi \mid y) \propto g(\theta, \phi), \quad \text{where } P(\theta, \phi \mid y) \text{ is a joint distribution} \\
& P(\phi \mid \theta, y) \propto P(\theta, \phi \mid y) \propto g(\theta, \phi)
\end{aligned}
$$
Gibbs sampling is to sample one parameter at a time based on other parameters' current values, then iterate.<br>
Algorithm<br>
1) initialize $\theta_{0}$, $\phi_{0}$<br>
2) for i = 1, 2, ..., m repeat<br>
&emsp;a) suing previous $\phi_{i-1}$ to draw $\theta_{i} \sim P(\theta \mid \phi_{i-1}, y)$<br>
&emsp;b) using $\theta_{i}$ to draw $\phi_{i} \sim P(\phi \mid \theta_{i}, y)$<br>
&emsp;c) get a pair ($\theta_{i}$, $\phi_{i}$)<br><br>

<b>Q</b>: <b>In what way does the Gibbs sampling algorithm simplify our task of updating multiple parameters in MCMC?</b><br>
A. It divides the process into updating one parameter at a time using (potentially convenient) full conditional distributions.<br>
B. The full conditional distribution eliminates the need to calculate a Metropolis-Hastings acceptance ratio.<br>
C. It updates all parameters at once, using a joint proposal distribution as part of a Metropolis-Hastings algorithm.<br>
D. Finding full conditional distributions requires less work than finding the unnormalized joint posterior distribution of all parameters.<br>
<b>Answer</b>: A.<br><br>
</p>

#### 2.5.2 Conditionally conjugate prior example with Normal likelihood
<p align="justify">
Suppose data follow a normal distribution with unknow mean and unknow variance
$$y_{i} \mid \mu, \sigma^{2} \stackrel{\text{i.i.d}}{\sim} N(\mu, \sigma^{2}), \quad i = 1, 2, ..., n$$
Normal prior for $\mu$
$$\mu \sim N(\mu_{0}, \sigma_{0}^{2})$$
Inverse-Gamma prior for $\sigma^{2}$
$$\sigma^{2} \sim \text{IG}(\nu_{0}, \beta_{0})$$
<b>If $\sigma^{2}$ is known constant, the normal distirbution is a conjugate prior for $\mu$; likewise, if $\mu$ is known, the Inverse-Gamma is conjugate prior for $\sigma^{2}$.</b> This gives us convenience for full conditional distribution in Gibbs sampling.<br>
So, posterior (joint distribution)
$$
\begin{aligned}
P(\mu, \sigma^{2} \mid y_{1}, ..., y_{n}) & \propto P(y_{1}, ..., y_{n} \mid \mu, \sigma^{2}) P(\mu) P(\sigma^{2}), \\
& \quad  \text{where } P(y_{1}, ..., y_{n} \mid \mu, \sigma^{2}) \text{ is likelihood for data} \\
& = [\prod_{i=1}^{n} N(y_{i} \mid \mu, \sigma^{2})] N(\mu \mid \mu_{0}, \sigma_{0}^{2}) \text{IG}(\sigma^{2} \mid \nu_{0}, \beta_{0}) \\
& = [\prod_{i=1}^{n} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp(-\frac{1}{2\sigma^{2}} (y_{i} - \mu)^{2})] \cdot \frac{1}{\sqrt{2 \pi \sigma_{0}^{2}}} \exp(-\frac{1}{2\sigma_{0}^{2}} (\mu - \mu_{0})^{2}) \\
& \quad \cdot \frac{\beta_{0}^{2}}{\Gamma(\nu_{0})} (\sigma^{2})^{-(\nu_{0} + 1)} \exp(-\frac{\beta_{0}}{\sigma^{2}}) \\
& \propto (\sigma^{2})^{-\frac{n}{2}} \exp(-\frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (y_{i}-\mu)^{2}) \\
& \quad \cdot \exp(-\frac{1}{2\sigma^{2}} (\mu - \mu_{0})^{2}) \cdot (\sigma^{2})^{-(\nu_{0} + 1)} \exp(-\frac{\beta_{0}}{\sigma^{2}})
\end{aligned}
$$
Then, full conditional distribution for $\mu$ assuming $\sigma^{2}$ is known
$$
\begin{aligned}
P(\mu \mid \sigma^{2}, y_{1}, ..., y_{n}) & \propto P(\mu, \sigma^{2} \mid y_{1}, ..., y_{n}) \\
& \propto \exp(-\frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (y_{i}-\mu)^{2}) \cdot \exp(-\frac{1}{2\sigma^{2}} (\mu - \mu_{0})^{2}) \\
& = \exp[-\frac{1}{2}(\frac{\sum_{i=1}^{n} (y_{i} - \mu)^{2}}{\sigma^{2}} + \frac{(\mu - \mu_{0})^{2}}{\sigma_{0}^{2}})] \\
& \propto N(\mu \mid \frac{n\frac{\bar{y}}{\sigma^{2}} + \frac{\mu_{0}}{\sigma_{0}^{2}}}{\frac{n}{\sigma^{2}} + \frac{1}{\sigma_{0}^{2}}}, \frac{1}{\frac{n}{\sigma^{2}} + \frac{1}{\sigma_{0}^{2}}})
\end{aligned}
$$
full conditional distribution for $\sigma^{2}$ assuming $\mu$ is known
$$
\begin{aligned}
P(\sigma^{2} \mid \mu, y_{1}, ..., y_{n}) & \propto P(\mu, \sigma^{2} \mid y_{1}, ..., y_{n}) \\
& \propto (\sigma^{2})^{-\frac{n}{2}} \exp(-\frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (y_{i}-\mu)^{2}) \cdot (\sigma^{2})^{-(\nu_{0} + 1)} \exp(-\frac{\beta_{0}}{\sigma^{2}}) \\
& = (\sigma^{2})^{-(\nu_{0} + \frac{n}{2} + 1)} \exp(-\frac{1}{\sigma^{2}} (\beta_{0} + \frac{\sum_{i=1}^{n} (y_{i} - \mu)^{2}}{2})) \\
& \propto \text{IG}(\sigma^{2} \mid \nu_{0} + \frac{n}{2}, \beta_{0} + \frac{\sum_{i=1}^{n}(y_{i} - \mu)^{2}}{2})
\end{aligned}
$$<br>

<b>Q</b>: If we implement the Gibbs sampler for the model described in this segment, <b>how do we complete an update for μ?</b><br>
A. Draw from the normal full conditional distribution for μ.<br>
B. Draw a candidate $\mu^{*}$ from a proposal distribution and use the full joint posterior for μ and $\sigma^{2}$ to evaluate the acceptance ratio.<br>
C. Draw a candidate $\mu^{*}$ from a proposal distribution and use the normal full conditional for μ to evaluate the acceptance ratio.<br>
D. Draw from the inverse-gamma full conditional distribution for μ.<br>
<b>Answer</b>: A.<br><br>
</p>

#### 2.5.3 Computing example with Normal likelihood
{% highlight R %}
# Normal distribution
update_mu = function(n, ybar, sig2, mu_0, sig2_0) {
  sig2_1 = 1.0 / (n / sig2 + 1.0 / sig2_0)
  mu_1 = sig2_1 * (n * ybar / sig2 + mu_0 / sig2_0)
  rnorm(n=1, mean=mu_1, sd=sqrt(sig2_1))
}

# Inverse-Gamma distribution
update_sig2 = function(n, y, mu, nu_0, beta_0) {
  nu_1 = nu_0 + n / 2.0
  sumsq = sum( (y - mu)^2 ) # vectorized
  beta_1 = beta_0 + sumsq / 2.0
  # rate for gamma is shape for inv-gamma
  out_gamma = rgamma(n=1, shape=nu_1, rate=beta_1)
  # reciprocal of a gamma random variable is distributed inv-gamma
  1.0 / out_gamma
}

# With functions for drawing from the full conditionals, we are ready
# to write a function to perform Gibbs sampling.
gibbs = function(y, n_iter, init, prior) {
  ybar = mean(y)
  n = length(y)
  
  ## initialize
  mu_out = numeric(n_iter)
  sig2_out = numeric(n_iter)
  
  mu_now = init$mu
  
  ## Gibbs sampler
  for (i in 1:n_iter) {
    sig2_now = update_sig2(n=n, y=y, mu=mu_now, nu_0=prior$nu_0,
                           beta_0=prior$beta_0)
    mu_now = update_mu(n=n, ybar=ybar, sig2=sig2_now, mu_0=prior$mu_0,
                       sig2_0=prior$sig2_0)
    
    sig2_out[i] = sig2_now
    mu_out[i] = mu_now
  }
  
  cbind(mu=mu_out, sig2=sig2_out)
}

y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)
ybar = mean(y)
n = length(y)

## prior
prior = list()
prior$mu_0 = 0.0
prior$sig2_0 = 1.0
prior$n_0 = 2.0 # prior effective sample size for sig2
prior$s2_0 = 1.0 # prior point estimate for sig2
# prior parameter for inverse-gamma
prior$nu_0 = prior$n_0 / 2.0
# prior parameter for inverse-gamma
prior$beta_0 = prior$n_0 * prior$s2_0 / 2.0

hist(y, freq=FALSE, xlim=c(-1.0, 3.0)) # histogram of the data
# prior for mu
curve(dnorm(x=x, mean=prior$mu_0, sd=sqrt(prior$sig2_0)),
      lty=2, add=TRUE)
points(y, rep(0,n), pch=1) # individual data points
points(ybar, 0, pch=19) # sample mean


# Finally, we can initialize and run the sampler!
set.seed(53)
init = list()
init$mu = 0.0
post = gibbs(y=y, n_iter=1e3, init=init, prior=prior)
head(post)

{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.5.4 Quiz
<p align="justify">
<b>1.</b><br>
Which of the following descriptions matches the process of Gibbs sampling for multiple random variables?<br>
A. Draw candidates for all variables simultaneously using a multivariate proposal distribution. Calculate the acceptance ratio α using the joint (unnormalized) density. Accept the candidates with probability min{1,α}. Repeat this step for many iterations.<br>
B. Cycle through the variables, drawing from a proposal distribution for each variable and accepting the candidate with probability equal to the ratio of the candidate draw to the old value of the variable. Repeat this cycle for many iterations.<br>
C. Draw candidates for all J variables simultaneously using a multivariate proposal distribution. For each variable, calculate the acceptance ratio $\alpha_{j}$ using the join (unnormalized) density. Accept each candidate with probability min{1, $\alpha_{j}$} for j=1,…,J. Repeat this cycle for many iterations.<br>
D. Cycle through the variables, drawing a sample from the full conditional distribution of each variable while substituting in the current values of all other variables. Repeat this cycle for many iterations.<br>
<b>Answer</b>: D.<br><br>

<b>2.</b><br>
Suppose we have a joint probability distribution for four variables, p(w,x,y,z). Which of the following expresses the full conditional distribution for variable x?<br>
A. P(x)<br>
B. P(x | y)<br>
C. P(w, y, z | x)<br>
D. P(x | w, y, z)<br>
<b>Answer</b>: D.<br><br>

<b>3.</b><br>
Suppose we have the following joint distribution for x,y, and z:
$$P(x, y, z) = 5e^{-5x} \mathbf{I}_{x \geq 0} \frac{\Gamma(z+3)}{\Gamma(x) \Gamma(3)} y^{x-1} (1 - y)^{2} \mathbf{I}_{0 < y < 1} \binom{10}{x} y^{x} (1-y)^{10-x}\mathbf{I}_{x \in \{1, ..., 10\}}$$
The density for the full conditional distribution of z is proportional to which of the following? Hint: The full conditional for z is proportional to the full joint distribution p(x,y,z) where x and y are just constants.
$$
\begin{aligned}
\text{A.} \quad & P(z \mid x, y) \propto 5e^{-5x} \mathbf{I}_{x \geq 0} \\
\text{B.} \quad & P(z \mid x, y) \propto y^{x-1} (1 - y)^{2} y^{x} (1-y)^{10-x}\mathbf{I}_{0 < y < 1} \\
\text{C.} \quad & P(z \mid x, y) \propto e^{-5x} \frac{\Gamma(z+3)}{\Gamma(x)} y^{x-1} \mathbf{I}_{x \geq 0} \\
\text{D.} \quad & P(z \mid x, y) \propto \binom{10}{x} y^{x} (1-y)^{10-x}\mathbf{I}_{x \in \{1, ..., 10\}}
\end{aligned}
$$
<b>Answer</b>: C.<br><br>

<b>4.</b><br>
The full conditional distribution in Question 3 is not a standard distribution that we can easily sample. Fortunately, it turns out that we can do a Metropolis-Hastings step inside our Gibbs sampler step for z. If we employ that strategy in a Gibbs sampler for y and z (always conditioning on x), then the algorithm would look like this:
</p>
{% highlight R %}
For iteration i in 1 to m, repeat:

  1. 
    a) Draw z* from a proposal distribution q.
    b) Calculate the acceptance ratio (alpha) using the full
        conditional distribution for z|x,y and the candidate
        distribution q, plugging in the previous iteration's
        value y_{i-1} for y.
    c) Accept the candidate with probability min{1,alpha} and
        set the value for z_i accordingly.

  2. ___________________________.

end.
{% endhighlight %}
<p align="justify">
What would go in step 2 to complete the Gibbs sampler?<br>
A. Draw $y_{i}$ from the marginal distribution p(y).<br>
B. Draw $y_{i}$ from the full conditional p(y∣x,z), plugging in the candidate $z^{*}$ for z.<br>
C. Draw $y_{i}$ from the full conditional p(y∣x,z), plugging in the value $z_{i}$ just drawn in step 1 for z.<br>
D. Draw $y_{i}$ from the full conditional p(y∣x,z), plugging in the previous iteration's value $z_{i-1}$ for z.<br>
<b>Answer</b>: C.<br><br>

<b>5.</b><br>
Suppose we have a joint probability distribution for three variables: p(x,y,z). Identify the algorithm to perform Gibbs sampling for all three variables.<br>
A.
</p>
{% highlight R %}
For iteration i in 1 to m, repeat:

  1. Draw candidates x*, y*, z* from a joint proposal
      distribution q.

  2. a) i) Calculate the acceptance ratio alpha_x using
            the full conditional p(x|y,z) and q, plugging in the
            candidates y*, z* for y and z.

        ii) Accept x* with probability min{1,alpha_x}
            and set x_i accordingly.

     b) i) Calculate the acceptance ratio alpha_y using
            the full conditional p(y|x,z) and q, plugging in x_i,
            z* for x and z.

        ii) Accept y* with probability min{1,alpha_y}
            and set y_i accordingly.

     c) i) Calculate the acceptance ratio alpha_z using
            the full conditional p(z|x,y) and q, plugging in x_i,
            y_i for x and y.

        ii) Accept z* with probability min{1,alpha_z}
            and set z_i accordingly.
end.
{% endhighlight %}
<p align="justify">
B.
</p>
{% highlight R %}
For iteration i in 1 to m, repeat:

  1. Draw x_i from the full conditional distribution for
      x|y,z, plugging in the previous iteration's values
      y_{i-1}, z_{i-1} for y and z.

  2. Draw y_i from the full conditional distribution for
      y|x,z, plugging in the previous iteration's values
      x_{i-1}, z_{i-1} for x and z.

  3. Draw z_i from the full conditional distribution for
      z|x,y, plugging in the previous iteration's values
      x_{i-1}, y_{i-1} for x and y.

end.
{% endhighlight %}
<p align="justify">
C.
</p>
{% highlight R %}
For iteration i in 1 to m, repeat:

  1. Draw candidates x*, y*, z* from a joint proposal
      distribution q.

  2. Calculate the acceptance ratio alpha using
      g(x,y,z) = p(x|y,z)p(y|x,z)p(z|x,y) and q.

  3. Accept the candidates with probability min{1,alpha}
      and set x_i, y_i, z_i accordingly.

end.
{% endhighlight %}
<p align="justify">
D.
</p>
{% highlight R %}
For iteration i in 1 to m, repeat:

  1. Draw x_i from the full conditional distribution for
      x|y,z, plugging in the previous iteration's values
      y_{i-1}, z_{i-1} for y and z.

  2. Draw y_i from the full conditional distribution for
      y|x,z, plugging in the previous iteration's value
      z_{i-1} for z and this iteration's value x_i for x.

  3. Draw z_i from the full conditional distribution for
      z|x,y, plugging in this iteration's values
      x_i, y_i for x and y.

end.
{% endhighlight %}
<p align="justify">
<b>Answer</b>: D.<br><br>

<b>6.</b><br>
For Questions 6 to 8, consider the example from the lesson where the data are percent change in total personnel since last year for n=10 companies. In our model with normal likelihood and unknown mean μ and unknown variance $\sigma^{2}$, we chose a normal prior for the mean and an inverse-gamma prior for the variance. What was the major advantage of selecting these particular priors?<br>
A. Each prior was conjugate in the case where the other parameter was known, causing the full conditional distributions to come from the same distribution families as the priors (and therefore easy to sample).<br>
B. Because these priors are conjugate for their respective parameters, they guarantee the smallest possible Monte Carlo standard error for posterior mean estimates.<br>
C. These priors allowed us to bypass MCMC, providing a joint conjugate posterior for μ and $\sigma^{2}$.<br>
D. Because these priors are conjugate for their respective parameters, they guarantee the most accurate posterior distribution possible for the given likelihood.<br>
<b>Answer</b>: A.<br><br>

<b>7.</b><br>
Suppose we repeat the analysis for n=6 companies in another industry and the data are:
</p>
{% highlight R %}
y = c(-0.2, -1.5, -5.3, 0.3, -0.8, -2.2)
{% endhighlight %}
<p align="justify">
Re-run the Gibbs sampler in R for these new data (5000 iterations using the same priors and initial values as in the Lesson) and report the posterior mean for μ. Round your answer to two decimal places.<br>
<b>Answer</b>: -0.98.<br><br>

<b>8.</b><br>
An industry expert is surprised by your results from Question 7 and insists that growth in this sector should be positive on average. To accommodate this expert's prior beliefs, you adjust the prior for μ to be normal with a mean 1.0 and variance 1.0. This is a fairly informative and optimistic prior (the prior probability that μ>0 is about 0.84). What happens to the posterior mean of μ? Re-run the analysis on the new data with this new prior. Again, use 5000 iterations and the same prior for $\sigma^{2}$ and initial values as before).<br>
A. The posterior mean for μ is less than −0.25, suggesting that despite the optimistic prior, the data strongly favor estimating growth to be negative in this industry.<br>
B. The posterior mean for μ is between −0.25 and 0.25, suggesting that the data are not as optimistic about growth as the prior, but we are inconclusive about whether growth is positive or negative.<br>
C. The posterior mean for μ is between 0.25 and 1.0, suggesting that the data are not informative enough to contradict this expert's opinion.<br>
D. The posterior mean for μ is above 1.0, suggesting that the optimistic prior was actually not optimistic enough.<br>
<b>Answer</b>: A.<br><br>
</p>

### 2.6 Assessing Convergence
#### 2.6.1 Autocorrelation
<p align="justify">
<b>Covariance and correlation</b><br>
In the previous course, we defined the expectation and variance of a random variable X as
$$\mu_{x} = E(X) = \int x \cdot f(x) dx, \quad \sigma_{x}^{2} = \text{Var}(X) = E[(X - \mu)^{2}]$$
where f is the probability density function (PDF) of X. If we have two random variables, we can also define the covariance between them as
$$\sigma_{xy} = \text{Cov}(X, Y) = E[(X - \mu_{x})(Y - \mu_{y})] = E[(X - E(X))(Y - E(Y))]$$
Notice that the variance of X is just the covariance of X with itself.
$$\text{Var}(X) = \text{Cov}(X, X)$$
Correlation between X and Y is defined as
$$\rho_{xy} = \text{Cor}(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X) \cdot \text{Var}(Y)}} = \frac{\sigma_{xy}}{\sqrt{\sigma_{x}^{2} \cdot \sigma_{y}^{2}}}$$
Correlation is often more useful than covariance because it is standardized to be between −1 and 1, which makes it interpretable as a standard measure of linear relationship between any two variables. We emphasize linear because there can be dependence between two variables that is not linear, and those kinds of dependence would not necessarily result in strong correlation. Hence, think of correlation as measuring the strength of the linear relationship between variables.
Two independent random variables will have covariance (and correlation) equal to 0. However, the converse is not true. If X and Y have correlation at or near 0, they are not necessarily independent.<br><br>

<b>Autocorrelation</b><br>
If we have a sequence of random variables $X_{1}$, $X_{2}$, . . . that are separated in time, as we did with the introduction to Markov chains, we can also think of the concept of autocorrelation, correlation of $X_{t}$ with some past or future variable $X_{t-l}$. Formally, it is defined as
$$\text{ACF}(X_{t}, X_{t-l}) = \frac{\text{Cov}(X_{t}, X_{t-l})}{\sqrt{\text{Var}(X_{t}) \cdot \text{Var}(X_{t-l})}}$$
If the sequence is stationary, so that the joint distribution of multiple Xs does not change with time shifts, then autocorrelation for two variables does not depend on the exact times t and t − l, but rather on the distance between them, l. That is why the autocorrelation plots in the lesson on convergence of MCMC calculate autocorrelation in terms of lags.<br><br>
</p>

#### 2.6.2 Quiz
<p align="justify">
<b>1.</b><br>
Why is it important to check your MCMC output for convergence before using the samples for inference?<br>
A. If the chain has not reached its stationary distribution (the target/posterior), your samples will not reflect that distribution.<br>
B. Pre-convergence MCMC samples are useless.<br>
C. You can cut your Monte Carlo error by a factor of two if you strategically select which samples to retain.<br>
D. Convergence diagnostics provide a guarantee that your inferences are accurate.<br>
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
Which of the following trace plots illustrates a chain that appears to have converged?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_6_2_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: D.<br><br>

<b>3.</b><br>
The trace plot below was generated by a random walk Metropolis sampler, where candidates were drawn from a normal proposal distribution with mean equal to the previous iteration's value, and a fixed variance. Based on this result, what action would you recommend taking next?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_6_2_2.png"/></center>
</p>
<p align="justify">
A. The step size of the proposals is too large. Increase the variance of the normal proposal distribution and re-run the chain.<br>
B. The step size of the proposals is too small. Increase the variance of the normal proposal distribution and re-run the chain.<br>
C. The step size of the proposals is too small. Decrease the variance of the normal proposal distribution and re-run the chain.<br>
D. The step size of the proposals is too large. Decrease the variance of the normal proposal distribution and re-run the chain.<br>
<b>Answer</b>: B.<br>
In other words, it takes too long for the chain to explore the posterior distribution. This is less of a problem if you run a very long chain, but it is best to use a more efficient proposal distribution if possible.<br><br>

<b>4.</b><br>
Suppose you have multiple MCMC chains from multiple initial values and they appear to traverse the same general area back and forth, but struggle from moderate (or high) autocorrelation. Suppose also that adjusting the proposal distribution q is not an option. Which of the following strategies is likely to help increase confidence in your Monte Carlo estimates?<br>
A. Add more chains from more initial values to see if that reduces autocorrelation.<br>
B. Run the chains for many more iterations and check for convergence on the larger time scale.<br>
C. Discard fewer burn-in samples to increase your Monte Carlo effective sample size.<br>
D. Retain only the 80% of samples closest to the maximum likelihood estimate.<br>
<b>Answer</b>: B.<br><br>

<b>5.</b><br>
Each of the following plots reports estimated autocorrelation from a MCMC chain with 10,000 iterations. Which will yield the lowest Monte Carlo effective sample size?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_6_2_3.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: A.<br>
High autocorrelation leads to low MCMC effective sample size.<br><br>

<b>6.</b><br>
The following trace plot shows four chains with distinct initial values. Of the choices given, what is the lowest number of samples you would comfortably recommend to discard as burn-in?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/BS/2_6_2_4.png"/></center>
</p>
<p align="justify">
A. 50 iterations<br>
B. 150 iterations<br>
C. 400 iterations<br>
D. 700 iterations<br>
<b>Answer</b>: C.<br><br>

<b>7.</b><br>
Suppose the Gelman and Rubin diagnostic computed from multiple chains reports a scale reduction factor much higher than 1.0, say 8.0. What is the recommended action?<br>
A. Use the samples for inference as this high scale reduction factor indicates convergence.<br>
B. Thin the chain by discarding every eighth sample.<br>
C. Continue running the chain for many more iterations.<br>
D. Discontinue use of the model, since there is little hope of reaching the stationary distribution.<br>
<b>Answer</b>: C.<br>
A high scale reduction factor indicates that the chains are not yet exploring the same space, so we need to provide them more iterations to converge.<br><br>

<b>8.</b><br>
Which of the following Monte Carlo statistics would require the largest MCMC effective sample size to estimate reliably? Assume the target distribution is unimodal (has only one peak).<br>
A. 15 percentile of the target distribution<br>
B. 97.5 percentile of the target distribution<br>
C. Mean of the target distribution<br>
D. Median of the target distribution<br>
<b>Answer</b>: B.<br>
The outer edges of the distribution are sampled less frequently and therefore susceptible to changes between simulations. The Raftery and Lewis diagnostic can help you decide how many iterations you need to reliably estimate outer quantiles of the target distribution.<br><br>
</p>

#### 2.6.3 Trace plots, autocorrelation
{% highlight R %}
set.seed(61)
post0 = mh(n=n, ybar=ybar, n_iter=10e3, mu_init=0.0, cand_sd=0.9)
coda::traceplot(as.mcmc(post0$mu[-c(1:500)]))

set.seed(61)
post1 = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=0.04)
coda::traceplot(as.mcmc(post1$mu[-c(1:500)]))

set.seed(61)
post2 = mh(n=n, ybar=ybar, n_iter=100e3, mu_init=0.0, cand_sd=0.04)
coda::traceplot(as.mcmc(post2$mu))

coda::autocorr.plot(as.mcmc(post0$mu))
coda::autocorr.diag(as.mcmc(post0$mu))
coda::autocorr.plot(as.mcmc(post1$mu))
coda::autocorr.diag(as.mcmc(post1$mu))

str(post2) # contains 100,000 iterations
coda::effectiveSize(as.mcmc(post2$mu)) # effective sample size of ~350
## thin out the samples until autocorrelation is essentially 0.
# This will leave you with approximately independent samples.
# The number of samples remaining is similar to the effective sample size.
coda::autocorr.plot(as.mcmc(post2$mu), lag.max=500)

# how far apart the iterations are for autocorrelation to be essentially 0.
thin_interval = 400
thin_indx = seq(from=thin_interval, to=length(post2$mu), by=thin_interval)
head(thin_indx)

post2mu_thin = post2$mu[thin_indx]
traceplot(as.mcmc(post2$mu))

traceplot(as.mcmc(post2mu_thin))

coda::autocorr.plot(as.mcmc(post2mu_thin), lag.max=10)

effectiveSize(as.mcmc(post2mu_thin))

length(post2mu_thin)

str(post0) # contains 10,000 iterations

coda::effectiveSize(as.mcmc(post0$mu)) # effective sample size of ~2,500

raftery.diag(as.mcmc(post0$mu))

raftery.diag(as.mcmc(post0$mu), q=0.005, r=0.001, s=0.95)
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.6.4 Multiple chains, burn-in, Gelman-Rubin diagnostic
{% highlight R %}
set.seed(62)
post3 = mh(n=n, ybar=ybar, n_iter=500, mu_init=10.0, cand_sd=0.3)
coda::traceplot(as.mcmc(post3$mu))

set.seed(61)

nsim = 500
post1 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=15.0, cand_sd=0.4)
post1$accpt
post2 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=-5.0, cand_sd=0.4)
post2$accpt
post3 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=7.0, cand_sd=0.1)
post3$accpt
post4 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=23.0, cand_sd=0.5)
post4$accpt
post5 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=-17.0, cand_sd=0.4)
post5$accpt
pmc = mcmc.list(as.mcmc(post1$mu), as.mcmc(post2$mu), 
                as.mcmc(post3$mu), as.mcmc(post4$mu), as.mcmc(post5$mu))
str(pmc)

coda::traceplot(pmc)

coda::gelman.diag(pmc)

coda::gelman.plot(pmc)

nburn = 1000 # remember to discard early iterations
post0$mu_keep = post0$mu[-c(1:1000)]
summary(as.mcmc(post0$mu_keep))
mean(post0$mu_keep > 1.0) # posterior probability that mu  > 1.0
{% endhighlight %}
<p align="justify">
<br>
</p>

### 2.7 Linear Regression
#### 2.7.1 Introduction to linear regression
<p align="justify">
$$y_{i} = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{k}x_{k} + \epsilon_{i}, \quad \epsilon \stackrel{\text{i.i.d}}{\sim} N(0, \sigma^{2}), \quad i = 1, ..., n$$
$$y_{i} \mid x, \beta y_{i} \stackrel{\text{i.i.d}}{\sim} N(\beta_{0} + \beta_{1}x_{1} + ... + \beta_{k}x_{k}, \sigma^{2})$$
The parameters comes from some prior
$$\beta_{0} \sim P(\beta_{0}), \quad \beta_{1} \sim P(\beta_{1}), ..., \quad \sigma^{2} \sim P(\sigma^{2})$$<br>

Laplace prior
$$P(\beta) = \frac{1}{2} e^{-\left | \beta \right |}$$<br>

<b>Q</b>: If we have k different predictor variables $x_{1}$, $x_{2}$, ..., $x_{k}$, what is the primary advantage of fitting joint a linear model (multiple regression, E(y)=$\beta_{0}$ + $\beta_{1}x_{1}$ + ... + $\beta_{k}x_{k}$) over fitting k simple linear regressions (E(y)=$\beta_{0}$ + $\beta_{j}x_{j}$), one for each predictor?<br>
A. Each coefficient in the multiple regression model accounts for the presence of the other predictors whereas the coefficients in the simple linear regressions don't.<br>
B. Fitting k simple linear regressions forces us to reconcile the k different estimates of the intercept $\beta_{0}$.<br>
C. We only have to perform one (instead of k) residual analysis.<br>
D. The coefficients in the multiple regression model can fit nonlinear relationships between the predictors and response which is not possible with simple linear regressions.<br>
<b>Answer</b>: A.<br><br>
</p>

#### 2.7.2 Setup in R
{% highlight R %}
library("car")
data("Leinhardt")
head(Leinhardt)
str(Leinhardt)
pairs(Leinhardt)
plot(infant ~ income, data=Leinhardt)
hist(Leinhardt$infant)
hist(Leinhardt$income)
Leinhardt$loginfant = log(Leinhardt$infant)
Leinhardt$logincome = log(Leinhardt$income)

plot(loginfant ~ logincome, data=Leinhardt)
lmod = lm(loginfant ~ logincome, data=Leinhardt)
summary(lmod)
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.7.3 JAGS model (linear regression)
{% highlight R %}
dat = na.omit(Leinhardt)
library("rjags")

mod1_string = " model {
    for (i in 1:n) {
y[i] ~ dnorm(mu[i], prec)
mu[i] = b[1] + b[2]*log_income[i] 
}

for (i in 1:2) {
b[i] ~ dnorm(0.0, 1.0/1.0e6)
}

prec ~ dgamma(5/2.0, 5*10.0/2.0)
sig2 = 1.0 / prec
sig = sqrt(sig2)
} "

set.seed(72)
data1_jags = list(y=dat$loginfant, n=nrow(dat), 
                  log_income=dat$logincome)

params1 = c("b", "sig")

inits1 = function() {
  inits = list("b"=rnorm(2,0.0,100.0),
               "prec"=rgamma(1,1.0,1.0))
}

mod1 = jags.model(textConnection(mod1_string),
                  data=data1_jags, inits=inits1,
                  n.chains=3)
update(mod1, 1000) # burn-in
# simulation
mod1_sim = coda.samples(model=mod1,
                        variable.names=params1,
                        n.iter=5000)
# combine multiple chains
mod1_csim = do.call(rbind, mod1_sim)
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.7.4 Model checking
{% highlight R %}
plot(mod1_sim)
gelman.diag(mod1_sim)
autocorr.diag(mod1_sim)
autocorr.plot(mod1_sim)
effectiveSize(mod1_sim)
summary(mod1_sim)
lmod0 = lm(infant ~ income, data=Leinhardt)
plot(resid(lmod0)) # to check independence (looks okay)
# to check for linearity, constant variance (looks bad)
plot(predict(lmod0), resid(lmod0))
# to check Normality assumption (we want this to be a straight line)
qqnorm(resid(lmod0))
# combine columns
X = cbind(rep(1.0, data1_jags$n), data1_jags$log_income)
head(X)

# posterior mean
(pm_params1 = colMeans(mod1_csim))

yhat1 = drop(X %*% pm_params1[1:2])
resid1 = data1_jags$y - yhat1
plot(resid1) # against data index

plot(yhat1, resid1) # against predicted values

qqnorm(resid1) # checking normality of residuals

# to compare with reference linear model
plot(predict(lmod), resid(lmod))

# which countries have the largest positive residuals?
rownames(dat)[order(resid1, decreasing=TRUE)[1:5]]
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.7.5 Quiz
<p align="justify">
<b>1.</b><br>
In a normal linear regression model with
$$E(u_{i}) = \beta_{0} + \beta_{1} x_{1, i} + \beta_{2} x_{2, i} + \beta_{3} x_{3, i}$$
which of the following gives the correct interpretation of $\beta_{2}$?<br>
A. while holding $x_{2, i}$ constant, the expectaion of $y_{i}$ is $\beta_{2}$<br>
B. while holding $x_{1, i}$ and $x_{3, i}$ constant, a one unit change in $x_{2, i}$, results in a $\beta_{2}$ change in the expectation of $y_{i}$<br>
C. when $x_{2, i}$ = 0, a one unit change in $x_{1, i}$ and $x_{3, i}$ results in a $\beta_{2}$ in $y_{i}$<br>
D. while holding $x_{1, i}$ and $x_{3, i}$ constant, a one unit change in $x_{2, i}$ results in a $\beta_{2}$ in $y_{i}$<br>
<b>Answer</b>: B.<br><br>

<b>2.</b><br>
Which of the following model specifications for $E(y_{i})$ is not a valid linear model?
$$
\begin{aligned}
\text{A.} \quad & \beta_{0} + \exp(\beta_{1}x_{1, i}) + \beta_{2}x_{2, i}^{2} \\
\text{B.} \quad & \beta_{0} + \beta_{1} x_{1, i} + \beta_{2} \frac{x_{1, i}}{x_{2, i}} \\
\text{C.} \quad & \beta_{0} + \beta_{1} \log(x_{1, i}) + \beta_{2} x_{2, i}^{2} \\
\text{D.} \quad & \beta_{0} + \beta_{1} \sin(2 \pi x_{1, i}) + \beta_{2} x_{2, i}
\end{aligned}
$$
<b>Answer</b>: A.<br>
This model is not linear in the coefficients. We are free to transform the predictors and the response, but the model itself must be linear.<br><br>

<b>3.</b><br>
Consider the Anscombe data set in R which can be accessed with the following code:
</p>
{% highlight R %}
library("car")  # load the 'car' package
data("Anscombe")  # load the data set
?Anscombe  # read a description of the data
head(Anscombe)  # look at the first few lines of the data
pairs(Anscombe)  # scatter plots for each pair of variables
{% endhighlight %}
<p align="justify">
Suppose we are interested in relating per-capita education expenditures to the other three variables. Which variable appears to have the strongest linear relationship with per-capita education expenditures?<br>
A. Proportion of population under age 18<br>
B. Per-capita income<br>
C. Proportion of population that is urban<br>
D. None of these variables appears to have a linear relationship with education expenditures.<br>
<b>Answer</b>: B.<br><br>

<b>4.</b><br>
Fit a reference (noninformative) Bayesian linear model to the Anscombe data with education expenditures as the response variable and include all three other variables as predictors. Use the lm function in R.What is the posterior mean estimate of the intercept in this model? Round your answer to one decimal place.<br>
<b>Answer</b>: -286.8.<br>
</p>
{% highlight R %}
lmod = lm(education ~ income + young + urban, data=Anscombe)
{% endhighlight %}
<p align="justify">
<br>
<b>5.</b><br>
In our reference analysis of the Anscombe data, the intercept is estimated to be negative. Does this parameter have a meaningful interpretation?<br>
A. Yes, it represents expected expenditures in a state with average income, average percent youth, and average percent urban.<br>
B. No, it represents expected expenditures in a state with 0 average income, 0 percent youth, and 0 percent urban which doesn't exist.<br>
C. No, there must be something wrong with the model because expenditures can never be negative.<br>
D. No, this model should not have an intercept term at all.<br>
<b>Answer</b>: B.<br><br>

<b>6.</b><br>
Use the code below to fit a linear regression model to the Anscombe data in JAGS. You will need to finish setting up and running the model.
</p>
{% highlight R %}
library("rjags")

mod_string = " model {
for (i in 1:length(education)) {
education[i] ~ dnorm(mu[i], prec)
mu[i] = b0 + b[1]*income[i] + b[2]*young[i] + b[3]*urban[i]
}

b0 ~ dnorm(0.0, 1.0/1.0e6)
for (i in 1:3) {
b[i] ~ dnorm(0.0, 1.0/1.0e6)
}

prec ~ dgamma(1.0/2.0, 1.0*1500.0/2.0)
## Initial guess of variance based on overall
## variance of education variable. Uses low prior
## effective sample size. Technically, this is not
## a true 'prior', but it is not very informative.
sig2 = 1.0 / prec
sig = sqrt(sig2)
} "

data_jags = as.list(Anscombe)
{% endhighlight %}
<p align="justify">
Before proceeding to inference, we should check our model. The first step is to check our MCMC chains. Do there appear to be any problems with the chains?<br>
A. Yes, there is very high autocorrelation for sig. We should help the chain for sig by fixing the initial value.<br>
B. Yes, there is very high autocorrelation among the coefficients. It would be good to run the chain for 100,000+ iterations to get reliable estimates.<br>
C. Yes, scale reduction factors are well above 1.0. The chains are not exploring the same distribution.<br>
D. No, a few thousand iterations will be sufficient for these chains.<br>
<b>Answer</b>: B.<br>
</p>
{% highlight R %}
params = c("b", "sig")
mod = jags.model(textConnection(mod_string),
                 data=data_jags,
                 n.chains=4)
update(mod, 1000) # burn-in
mod_sim = coda.samples(model=mod,
                       variable.names=params,
                       n.iter=5000)
mod_csim = do.call(rbind, mod_sim)
plot(mod_sim)
gelman.diag(mod_sim)
autocorr.diag(mod_sim)
autocorr.plot(mod_sim)
effectiveSize(mod_sim)
{% endhighlight %}
<p align="justify">
<br>

<b>7.</b><br>
Which of the following is not a condition we can check using a residual plot with predicted values on the x-axis and residuals on the y-axis?<br>
A. Independence of the observations<br>
B. Presence of outliers<br>
C. Constant error variance<br>
D. Linearity of the relationship between predictors and the response<br>
<b>Answer</b>: A.<br>
</p>
{% highlight R %}
plot(resid(lmod))
plot(predict(lmod), resid(lmod))
qqnorm(resid(lmod))
{% endhighlight %}
<p align="justify">
One way to check this assumption is by plotting predicted values against the data index. In the Anscombe data, we could check to see if residuals are more similar for states that are geographically close than for states that are not geographically close. If that is true, there may be spatial correlation in the data.<br><br>

<b>8.</b><br>
Check the residual plot described in Question 7 for the Anscombe data. Since the estimates of the coefficients in the reference model are very close to those in the JAGS model, we will just look at the residuals of the reference model. This plot is the first that appears when you run the following code:
</p>
{% highlight R %}
plot(mod_lm)
# here mod_lm is the object saved when you run lm()
{% endhighlight %}
<p align="justify">
Do there appear to be any issues with this fit?<br>
A. Yes, the observations appear not to be independent.<br>
B. No, this plot raises no concerns.<br>
C. Yes, there is a curved pattern or shape to the residuals, indicating a nonlinear relationship between the variables.<br>
D. Yes, there are a few extreme outliers.<br>
E. Yes, the error variability appears to increase as predicted values increase.<br>
<b>Answer</b>: E.<br><br>
</p>

#### 2.7.6 Alternative models
{% highlight R %}
library("rjags")

mod2_string = " model {
for (i in 1:length(y)) {
y[i] ~ dnorm(mu[i], prec)
mu[i] = b[1] + b[2]*log_income[i] + b[3]*is_oil[i]
}

for (i in 1:3) {
b[i] ~ dnorm(0.0, 1.0/1.0e6)
}

prec ~ dgamma(5/2.0, 5*10.0/2.0)
sig = sqrt( 1.0 / prec )
} "


set.seed(73)
data2_jags = list(y=dat$loginfant,
                  log_income=dat$logincome,
                  is_oil=as.numeric(dat$oil=="yes"))
data2_jags$is_oil

params2 = c("b", "sig")

inits2 = function() {
  inits = list("b"=rnorm(3,0.0,100.0),
               "prec"=rgamma(1,1.0,1.0))
}

mod2 = jags.model(textConnection(mod2_string),
                  data=data2_jags, inits=inits2,
                  n.chains=3)
update(mod2, 1e3) # burn-in

mod2_sim = coda.samples(model=mod2,
                        variable.names=params2,
                        n.iter=5e3)

# combine multiple chains
mod2_csim = as.mcmc(do.call(rbind, mod2_sim))

plot(mod2_sim)
gelman.diag(mod2_sim)
autocorr.diag(mod2_sim)
autocorr.plot(mod2_sim)

effectiveSize(mod2_sim)
summary(mod2_sim)

X2 = cbind(rep(1.0, data1_jags$n),
           data2_jags$log_income,
           data2_jags$is_oil)
head(X2)

(pm_params2 = colMeans(mod2_csim)) # posterior mean

yhat2 = drop(X2 %*% pm_params2[1:3])
resid2 = data2_jags$y - yhat2
plot(resid2) # against data index

plot(yhat2, resid2) # against predicted values

plot(yhat1, resid1) # residuals from the first model

sd(resid2) # standard deviation of residuals

mod3_string = " model {
    for (i in 1:length(y)) {
y[i] ~ dt( mu[i], tau, df )
mu[i] = b[1] + b[2]*log_income[i] + b[3]*is_oil[i]
}

for (i in 1:3) {
b[i] ~ dnorm(0.0, 1.0/1.0e6)
}

# we want degrees of freedom > 2 to guarantee
# existence of mean and variance
df = nu + 2.0
nu ~ dexp(1.0)

# tau is close to, but not equal to the precision
tau ~ dgamma(5/2.0, 5*10.0/2.0)
# standard deviation of errors
sig = sqrt( 1.0 / tau * df / (df - 2.0) )
} "
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.7.7 Deviance information criterion (DIC)
{% highlight R %}
# a small deviance means a high likelihood
# a better fitted model has a lower DIC (penalized deviance)
dic.samples(mod1, n.iter=1e3)
dic.samples(mod2, n.iter=1e3)
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.7.8 Quiz
<p align="justify">
<b>1.</b><br>
What is the primary interpretation of the penalty term in the deviance information criterion (DIC)?<br>
A. It gives an estimate of how much your mean squared error would increase for each additional parameter estimated.<br>
B. It estimates the optimal number of predictor variables (covariates) to include in the model.<br>
C. It gives an effective number of parameters estimated in the model.<br>
D. It penalizes overly simple models.<br>
<b>Answer</b>: C.<br>
It penalizes overly complicated models which fit this particular data set well, but may fail to generalize. This penalty will be particularly useful for hierarchical models.<br><br>

<b>2.</b><br>
DIC is a helpful tool for selecting among competing models. Which of the following changes to a linear model is not appropriate to evaluate with DIC?<br>
A. Adding or removing candidate covariates (predictors)<br>
B. Transformation of covariates (predictors)<br>
C. Choice of distribution for the likelihood<br>
D. Minor changes to the prior distributions<br>
<b>Answer</b>: D.<br>
If we optimize the model with respect to the prior, we might as well have not used priors. This practice can lead to inflated confidence and misleading results. One exception is if we use a completely different class of priors or prior structure that has a specific purpose, like variable selection. We will explore this in the next lesson.<br><br>

<b>3.</b><br>
Although the residual analysis of the Anscombe data showed no major problem that we will pursue, it is still worthwhile to compare some competing models. First, calculate and report the DIC for the original model (that you fit for the previous quiz). Round your answer to the nearest whole number. Hint: Use the dic.samples function in the rjags package and use a large number of samples (around 100,000) for a reliable answer. DIC is the last number reported with the title "Penalized deviance."<br>
<b>Answer</b>: 486.<br>
</p>
{% highlight R %}
dic.samples(mod, n.iter=100000)
{% endhighlight %}
<p align="justify">
<br>

<b>4.</b><br>
We will consider two alternative models for the Anscombe data. Because income and urban may be more highly correlated with each other than with education, and since urban was less significant than income in our models so far, we'll consider dropping it (we'll discuss correlated covariates more in the next lesson). The two alternative models we will try are based on these adjustments:<br>
1) Remove the term in the linear model for urban.<br>
2) In addition to dropping urban, add an interaction term $\beta_{3}$×income×youth. Fit both models in JAGS and calculate the DIC for each. If predictive performance is our criterion, which model would you conclude performs best?<br>
A. The DIC is indistinguishable among the three models. We cannot clearly identify a preferred model.<br>
B. The DIC is lowest for the original model with all covariates. This is our preferred model.<br>
C. The DIC is lowest for the second model without the urban covariate. This is our preferred model.<br>
D. The DIC is lowest for the third model with the interaction term. This is our preferred model.<br>
<b>Answer</b>: B.<br><br>

<b>5.</b><br>
Using the model favored by the DIC, obtain a Monte Carlo estimate of the posterior probability that the coefficient for income is positive (greater than 0.0). Round your answer to two decimal places.<br>
<b>Answer</b>: 1.0.<br>
</p>
{% highlight R %}
mean(mod_csim[:, 1] > 0)
{% endhighlight %}
<p align="justify">
<br>

<b>6.</b><br>
Which of the following accurately summarizes our conclusions based on the model favored by the DIC?<br>
A. Increases in per-capita income and percent urban are associated with increases in mean per-capita education expenditures. Increases in percent youth are associated with decreases in mean per-capita education expenditures.<br>
B. Increases in per-capita income and percent youth are associated with decreases in mean per-capita education expenditures. Increases in percent urban are associated with increases in mean per-capita education expenditures.<br>
C. Increases in per-capita income and percent youth are associated with decreases in mean per-capita education expenditures. Increases in percent urban are irrelevant.<br>
D. Increases in per-capita income and percent youth are associated with increases in mean per-capita education expenditures. Increases in percent urban are associated with decreases in mean per-capita education expenditures.<br>
<b>Answer</b>: D.<br><br>
</p>

### 2.8 ANOVA
#### 2.8.1 Introduction to ANOVA
<p align="justify">
Anova is used when we have categorical explanatory variables so that the observations belong to groups.<br><br>

<b>Q</b>: ANOVA is an appropriate model in which of the following scenarios?<br>
A. We fit a linear model where the response is income and the only predictor is level of education, which is one of: no secondary education diploma, secondary education diploma, or college degree.<br>
B. We fit a linear model where the response is income and there are two predictors: years of experience and income for the previous year.<br>
C. We fit a linear model where the response is income and the only predictor is the income for the previous year.<br>
D. We fit a linear model where the response is income and the only predictor is age in years.<br>
<b>Answer</b>: A.<br><br>
</p>

#### 2.8.2 One way model using JAGS
{% highlight R %}
data("PlantGrowth")
head(PlantGrowth)

boxplot(weight ~ group, data=PlantGrowth)

lmod = lm(weight ~ group, data=PlantGrowth)
summary(lmod)

anova(lmod)

plot(lmod) # for graphical residual analysis

library("rjags")

mod_string = " model {
    for (i in 1:length(y)) {
y[i] ~ dnorm(mu[grp[i]], prec)
}

for (j in 1:3) {
mu[j] ~ dnorm(0.0, 1.0/1.0e6)
}

prec ~ dgamma(5/2.0, 5*1.0/2.0)
sig = sqrt( 1.0 / prec )
} "

set.seed(82)
str(PlantGrowth)
data_jags = list(y=PlantGrowth$weight, 
                 grp=as.numeric(PlantGrowth$group))

params = c("mu", "sig")

inits = function() {
  inits = list("mu"=rnorm(3,0.0,100.0),
               "prec"=rgamma(1,1.0,1.0))
}

mod = jags.model(textConnection(mod_string),
                 data=data_jags,
                 inits=inits, n.chains=3)
update(mod, 1e3)

mod_sim = coda.samples(model=mod,
                       variable.names=params,
                       n.iter=5e3)
mod_csim = as.mcmc(do.call(rbind, mod_sim)) # combined chains

# Trace plot
plot(mod_sim)
gelman.diag(mod_sim)
autocorr.diag(mod_sim)
effectiveSize(mod_sim)

(pm_params = colMeans(mod_csim))

yhat = pm_params[1:3][data_jags$grp]
resid = data_jags$y - yhat
plot(resid)

plot(yhat, resid)

summary(mod_sim)

HPDinterval(mod_csim)

mean(mod_csim[,3] > mod_csim[,1])

mean(mod_csim[,3] > 1.1*mod_csim[,1])
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.8.3 Quiz
<p align="justify">
<b>1.</b><br>
Which of the following variables qualifies as a "factor" variable?<br>
A. Patient age in years<br>
B. Treatment with either an experimental drug or a control placebo<br>
C. Weight of a patient reported in kilograms<br>
D. Pre-treatment temperature reported in degrees Celsius<br>
<b>Answer</b>: B.<br><br>

<b>2.</b><br>
In an ANOVA model for a single factor with four levels, there are multiple ways we can parameterize our model for E(y). These include the cell means model or a linear model with a baseline mean and adjustments for different levels. Regardless of the model chosen, what is the maximum number of parameters we use to relate this factor with E(y) in a linear model and still be able to uniquely identify the parameters?<br>
<b>Answer</b>: 4.<br><br>

<b>3. For Questions 3-8, refer to the plant growth analysis from the lesson</b><br>
e-fit the JAGS model on plant growth from the lesson with a separate variance for each of the three groups. To do so, modify the model code to index the precision in the normal likelihood by group, just as we did with the mean. Use the same priors as the original model (except in this case it will be three independent priors for the variances).<br>
Compare the estimates between the original lesson model and this model with the summary function. Notice that the posterior means for the three μ parameters are essentially unchanged. However, the posterior variability for these parameters has changed. The posterior for which group's mean was most affected by fitting separate group variances?<br>
A. Group 1: control<br>
B. Group 2: treatment 1<br>
C. Group 3: treatment 2<br>
D. The effect on the marginal posterior was the same for all three groups.<br>
<b>Answer</b>: B.<br><br>

<b>4.</b><br>
Compute the deviance information criterion (DIC) for each of the two models and save the results as objects dic1 (for the original model) and dic2 (for the new model). Wha is the difference: DIC1 - DIC2? Hint: You can compute this directly with the following code: dic1−dic2.<br>
<b>Answer</b>: -3.88979.<br><br>

<b>5.</b><br>
Based on the DIC calculations for these competing models, what should we conclude?<br>
A. The DIC is lower for the original model, indicating preference for the model with one common variance across groups.<br>
B. The DIC is higher for the original model, indicating preference for the model with one common variance across groups.<br>
C. The DIC is higher for the new model, indicating preference for the model with separate variances across groups.<br>
D. The DIC is lower for the new model, indicating preference for the model with separate variances across groups.<br>
<b>Answer</b>: A.<br><br>

<b>6.</b><br>
Use the original model (single variance) to calculate a 95% interval of highest posterior density (HPD) for $\mu_{3}$-$\mu_{1}$. Which of the following is closest to this interval?<br>
A. (-0.14, 1.13)<br>
B. (-0.20, 1.19)<br>
C. (0.22, 1.49)<br>
D. (-1.01, 0.25)<br>
<b>Answer</b>: A.<br><br>

<b>7.</b><br>
What is the correct interpretation of $\mu_{3}$-$\mu_{1}$ in the context of the plant growth analysis?<br>
A. It is the difference in plant weight between treatment 2 and control.<br>
B. It is the effect (change) of treatment 2 with respect to the control in mean plant weight.<br>
C. It is the effect (change) of treatment 2 with respect to the control in plant weight.<br>
D. It is the mean range of plant weight across the three treatment groups.<br>
<b>Answer</b>: B.<br><br>

<b>8.</b><br>
The linear model with a baseline mean and group effects is the default in R. However, we can also fit the cell means model in R using the following code:
</p>
{% highlight R %}
mod_cm = lm(weight ~ -1 + group, data=PlantGrowth)
summary(mod_cm)
{% endhighlight %}
<p align="justify">
where the −1 in the model formula tells R to drop the intercept. Because we used fairly noninformative priors for the μ parameters in the analysis with JAGS, the results are very similar. In addition to allowing different prior specifications, what is one advantage of posterior sampling with JAGS over fitting the reference model in R?<br>
A. We can estimate the proportion of the variation in plant weight attributable to the treatment group assignment.<br>
B. We can use the posterior samples to obtain simulated posterior distributions of any function of the parameters that may interest us (e.g., $\mu_{3}$-$\mu_{1}$.<br>
C. We can obtain posterior standard deviations (standard errors) for each mean (or coefficient).<br>
D. We can obtain posterior mode estimates for each mean (or coefficient).<br>
<b>Answer</b>: B.
</p>
{% highlight R %}
mod2_string = " model {
    for (i in 1:length(y)) {
      y[i] ~ dnorm(mu[grp[i]], prec[grp[i]])
    }
    for (j in 1:3) {
      mu[j] ~ dnorm(0.0, 1.0/1.0e6)
      prec[j] ~ dgamma(5/2.0, 5*1.0/2.0)
      sig[j] = sqrt( 1.0 / prec[j] )
    }
} "

inits2 = function() {
  inits = list("mu"=rnorm(3,0.0,100.0), "prec"=rgamma(3,1.0,1.0))
}

mod2 = jags.model(textConnection(mod2_string),
                  data=data_jags,
                  inits=inits2, n.chains=3)
update(mod2, 1e3)

mod2_sim = coda.samples(model=mod2,
                        variable.names=params,
                        n.iter=5e3)
mod2_csim = as.mcmc(do.call(rbind, mod2_sim)) # combined chains

dic1 = dic.samples(mod, n.iter=100000)
dic2 = dic.samples(mod2, n.iter=100000)
{% endhighlight %}
<p align="justify">
<br>
</p>

### 2.9 Logistic Regression
#### 2.9.1 Introduction to logistic regression
<p align="justify">
$$y_{i} \mid \phi_{i} \stackrel{\text{i.i.d}}{\sim} \text{Bern}(\phi_{i}), \quad i = 1, 2, ... n$$
$$\text{logit}(\phi_{i}) = \log(\frac{\phi_{i}}{1 - \phi_{i}}) = \beta_{0} + \beta_{1} x_{1i}$$
$$E(y_{i}) = \phi_{i} = \frac{e^{\beta_{0} + \beta_{1} x_{i}}}{1 + e^{\beta_{0} + \beta_{1} x_{i}}} = \frac{1}{1 + e^{-(\beta_{0} + \beta_{1}x_{i})}}$$<br>
</p>

#### 2.9.2 JAGS model (logistic regression)
{% highlight R %}
library("boot")
data("urine")
head(urine)
dat = na.omit(urine)
pairs(dat)
library("corrplot")
Cor = cor(dat)
corrplot(Cor, type="upper", method="ellipse", tl.pos="d")
corrplot(Cor, type="lower", method="number", col="black", 
         add=TRUE, diag=FALSE, tl.pos="n", cl.pos="n")
X = scale(dat[,-1], center=TRUE, scale=TRUE)
head(X[,"gravity"])
colMeans(X)
apply(X, 2, sd)
ddexp = function(x, mu, tau) {
  0.5*tau*exp(-tau*abs(x-mu)) 
}
# double exponential distribution
curve(ddexp(x, mu=0.0, tau=1.0), from=-5.0, to=5.0,
      ylab="density",
      main="Double exponential\ndistribution")
# normal distribution
curve(dnorm(x, mean=0.0, sd=1.0), from=-5.0, to=5.0,
      lty=2, add=TRUE)
legend("topright", legend=c("double exponential", "normal"),
       lty=c(1,2), bty="n")
library("rjags")
mod1_string = " model {
    for (i in 1:length(y)) {
y[i] ~ dbern(p[i])
logit(p[i]) = int + b[1]*gravity[i] + b[2]*ph[i] +
              b[3]*osmo[i] + b[4]*cond[i] +
              b[5]*urea[i] + b[6]*calc[i]
}
int ~ dnorm(0.0, 1.0/25.0)
for (j in 1:6) {
b[j] ~ ddexp(0.0, sqrt(2.0)) # has variance 1.0
}
} "
set.seed(92)
head(X)
data_jags = list(y=dat$r, gravity=X[,"gravity"],
                 ph=X[,"ph"], osmo=X[,"osmo"],
                 cond=X[,"cond"], urea=X[,"urea"],
                 calc=X[,"calc"])
params = c("int", "b")
mod1 = jags.model(textConnection(mod1_string),
                  data=data_jags, n.chains=3)
update(mod1, 1e3)
mod1_sim = coda.samples(model=mod1,
                        variable.names=params,
                        n.iter=5e3)
mod1_csim = as.mcmc(do.call(rbind, mod1_sim))
## convergence diagnostics
plot(mod1_sim, ask=TRUE)
gelman.diag(mod1_sim)
autocorr.diag(mod1_sim)
autocorr.plot(mod1_sim)
effectiveSize(mod1_sim)
## calculate DIC
dic1 = dic.samples(mod1, n.iter=1e3)
summary(mod1_sim)
par(mfrow=c(3,2))
densplot(mod1_csim[,1:6], xlim=c(-3.0, 3.0))
colnames(X) # variable names
mod2_string = " model {
    for (i in 1:length(y)) {
y[i] ~ dbern(p[i])
logit(p[i]) = int + b[1]*gravity[i] + b[2]*cond[i] + b[3]*calc[i]
}
int ~ dnorm(0.0, 1.0/25.0)
for (j in 1:3) {
b[j] ~ dnorm(0.0, 1.0/25.0) # noninformative for logistic regression
}
} "
mod2 = jags.model(textConnection(mod2_string), data=data_jags, n.chains=3)
update(mod2, 1e3)
mod2_sim = coda.samples(model=mod2,
                        variable.names=params,
                        n.iter=5e3)
mod2_csim = as.mcmc(do.call(rbind, mod2_sim))
plot(mod2_sim, ask=TRUE)
gelman.diag(mod2_sim)
autocorr.diag(mod2_sim)
autocorr.plot(mod2_sim)
effectiveSize(mod2_sim)
dic2 = dic.samples(mod2, n.iter=1e3)
dic1
dic2
summary(mod2_sim)
HPDinterval(mod2_csim)
par(mfrow=c(3,1))
densplot(mod2_csim[,1:3], xlim=c(-3.0, 3.0))
colnames(X)[c(1,4,6)] # variable names
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.9.3 Prediction
{% highlight R %}
(pm_coef = colMeans(mod2_csim))
pm_Xb = pm_coef["int"] + X[,c(1,4,6)] %*% pm_coef[1:3]
phat = 1.0 / (1.0 + exp(-pm_Xb))
head(phat)
plot(phat, jitter(dat$r))
(tab0.5 = table(phat > 0.5, data_jags$y))
sum(diag(tab0.5)) / sum(tab0.5)
(tab0.3 = table(phat > 0.3, data_jags$y))
sum(diag(tab0.3)) / sum(tab0.3)
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 2.9.4 Quiz
<p align="justify">
<b>1.</b><br>
What is the advantage of using a link function such as the logit transform for logistic regression?<br>
A. It ensures that the β coefficients lie between 0 and 1 for all values of predictors x.<br>
B. It ensures that $\beta_{0} + \beta_{1}x_{1} + ... + \beta_{k}x_{k}$ is between 0 and 1 using log transformations of the β coefficients.<br>
C. It makes the β coefficients interpretable directly as probabilities.<br>
D. It ensures that the success probability (E(y) if y is Bernoulli) is between 0 and 1 without requiring any constraints on the x variables or the β coefficients.<br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
Logistic regression works with binomial likelihoods in addition to Bernoulli likelihoods. If the response $y_{1}$ is a number of successes in $n_{1}$ independent trials each with $\phi_{1}$ success probability, we can still model $\phi_{1}$ with a linear model using the logit transformation.<br>
As an example, consider the OME data in the MASS package in R. The data consist of experimental results from tests of auditory perception in children. Under varying conditions and for multiple trials under each condition, children either correctly or incorrectly identified the source of changing signals.<br>
Although the independence of the trails and results are questionable, we'll try fitting a logistic regression to these data. First, we'll explore the relationships briefly with the following code:
</p>
{% highlight R %}
library("MASS")
data("OME")
?OME # background on the data
head(OME)

any(is.na(OME)) # check for missing values
# manually remove OME missing values identified with "N/A"
dat = subset(OME, OME != "N/A")
dat$OME = factor(dat$OME)
str(dat)

plot(dat$Age, dat$Correct / dat$Trials )
plot(dat$OME, dat$Correct / dat$Trials )
plot(dat$Loud, dat$Correct / dat$Trials )
plot(dat$Noise, dat$Correct / dat$Trials )

{% endhighlight %}
<p align="justify">
We are interested how these variables relate to the probability of successfully identifying the source of changes in sound. Of these four variables, which appears to have the weakest association with the probability of success?<br>
A. Age in months<br>
B. OME: degree of otitis media with effusion (low or high)<br>
C. Loudness of stimulus in decibels<br>
D. Noise: stimulus type (coherent or incoherent)<br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
Next, we'll fit a reference logistic regression model with noninformative prior in R. We can do this with the glm function, providing the model formula as with the usual lm, except now the response is the observed proportion of correct responses. We must also indicate how many trials were run for each experiment using the weights argument.
</p>
{% highlight R %}
mod_glm = glm(Correct/Trials ~ Age + OME + Loud + Noise, data=dat, weights=Trials, family="binomial")
summary(mod_glm)
{% endhighlight %}
<p align="justify">
To get an idea of how the model fits, we can create residual (using a special type of residual for non-normal likelihoods) and in-sample prediction plots.
</p>
{% highlight R %}
plot(residuals(mod_glm, type="deviance"))
plot(fitted(mod_glm), dat$Correct/dat$Trials)
{% endhighlight %}
<p align="justify">
It appears from the second plot that the model is not very precise (some model predictions were far from the observed proportion of correct responses). Nevertheless, it can be informative about the relationships among the variables. Report the posterior mode estimate of the coefficient for low OME. Round your answer to two decimal places.<br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
Next, we will fit a similar model in JAGS. To make the results comparable to those of the reference model, we will use the same configuration of covariates. We can extract this information from the reference model using model.matrix.
</p>
{% highlight R %}
X = model.matrix(mod_glm)[,-1] # -1 removes the column of 1s for the intercept
head(X)
{% endhighlight %}
<p align="justify">
The data include categorical covariates which R codes as dummy variables (as with ANOVA). Hence we have an indicator variable for whether OME is at its low level and another indicating whether the Noise is incoherent. The intercept is then associated with this baseline group. Ignoring the continuous variables Age and Loud, what are the characteristics of this baseline group?<br>
A. Children with low OME exposed to coherent sound.<br>
B. Children with high OME exposed to incoherent sound.<br>
C. Children with low OME exposed to incoherent sound.<br>
D. Children with high OME exposed to coherent sound.<br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
Now complete the following code (as well as the code from previous questions) to fit the JAGS model with the fairly noninformative priors given. Use three chains with at least 5,000 iterations in each.
</p>
{% highlight R %}
mod_string = " model {
	for (i in 1:length(y)) {
		y[i] ~ dbin(phi[i], n[i])
		logit(phi[i]) = b0 + b[1]*Age[i] + b[2]*OMElow[i] +
                    b[3]*Loud[i] + b[4]*Noiseincoherent[i]
	}
	
	b0 ~ dnorm(0.0, 1.0/5.0^2)
	for (j in 1:4) {
		b[j] ~ dnorm(0.0, 1.0/4.0^2)
	}
	
} "
data_jags = as.list(as.data.frame(X))
# this will not work if there are missing values in dat (because they
# would be ignored by model.matrix). Always make sure that the data are
# accurately pre-processed for JAGS.
data_jags$y = dat$Correct
data_jags$n = dat$Trials
# make sure that all variables have the same number of observations (712).
str(data_jags)
{% endhighlight %}
<p align="justify">
Because there are many data points, the MCMC will take some time to run. Before analyzing the results, perform some MCMC diagnostic checks. What does the Raftery and Lewis diagnostic (raftery.diag()) suggest about these chains?<br>
A. The scale reduction factor for many variables is large (>5.0), indicating that the different chains are not exploring the same space yet. We need to run a longer burn-in period.<br>
B. The dependence factor for many of the variables is large (>5.0), indicating weak autocorrelation in the chains. We would not require a large number of iterations to reliably produce 95% probability intervals for the parameters.<br>
C. The dependence factor for many of the variables is large (>5.0), indicating strong autocorrelation in the chains. We would require a large number of iterations to reliably produce 95% probability intervals for the parameters.<br>
D. The scale reduction factor for many variables is large (>5.0), indicating that the different chains are exploring the same space. We have used a sufficient burn-in time.<br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
Although OMElow is the predictor with weakest statistical association to probability of correct responses, the posterior probability that its coefficient $\beta_{2}$ is negative is still greater than 0.9. How do we interpret this (most likely) negative coefficient in the context of our models?<br>
A. While holding all other predictors constant, low OME is associated with a decrease of magnitude |$\beta_{2}$| in the probability of correct responses while high OME is associated with an increase |$\beta_{2}$|<br>
B. While holding all other predictors constant, low OME is associated with a higher probability of correct responses than high OME.<br>
C. While holding all other predictors constant, low OME is associated with a lower probability of correct responses than high OME.<br>
D. While holding all other predictors constant, low OME is associated with an increase of magnitude |$\beta_{2}$| in the probability of correct responses while high OME is associated with a decrease of |$\beta_{2}$|<br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
Using the posterior mean estimates of the model coefficients, create a point estimate of the probability of correct responses for a child of age 60 months, with high OME, using a coherent stimulus of 50 decibels. Round your answer to two decimal places. Hint: First calculate the linear part by multiplying the variables by the coefficients and adding them up (call this xb). Once you have that, apply the inverse of the link function to transform it into a probability estimate.<br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
Use the posterior mean estimates of the model coefficients to create point estimates of the probability of correct responses for each observation in the original data. To do this, follow the steps outlined in the lesson to create a vector of these probabilities called phat (using our notation from this quiz, it would be $\hat{\phi}$). Once you have phat, calculate the proportion of in-sample observations that are correctly classified according to the following criterion: the model prediction and observed correct response rate are either both higher than 0.7 or both lower than 0.7. Round your answer to two decimal places. Hint: Use the following code:
</p>
{% highlight R %}
(tab0.7 = table(phat > 0.7, (dat$Correct / dat$Trials) > 0.7))
sum(diag(tab0.7)) / sum(tab0.7)
{% endhighlight %}
<p align="justify">
<b>Answer</b>: .<br><br>
</p>

#### 2.9.6 Reading: Multiple factor ANOVA
<p align="justify">

</p>

### 2.10 Poisson Regression
#### 2.10.1 LectureIntroduction to Poisson regression
<p align="justify">

</p>

#### 2.10.2 JAGS model (Poisson regression)
<p align="justify">

</p>

#### 2.10.3 Predictive distributions
<p align="justify">

</p>

#### 2.10.4 Quiz
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>

#### 2.10.5 Reading: Prior sensitivity analysis
<p align="justify">

</p>

#### 2.10.6 Reading: Code for Lesson 10
{% highlight R %}

{% endhighlight %}
<p align="justify">

</p>

### 2.11 Hierarchical Modeling
#### 2.11.1 Correlated data
<p align="justify">

</p>

#### 2.11.2 Reading: Normal hierarchical model
<p align="justify">

</p>

#### 2.11.3 Prior predictive simulation
<p align="justify">

</p>

#### 2.11.4 JAGS model and model checking (hierarchical modeling)
<p align="justify">

</p>

#### 2.11.5 Posterior predictive simulation
<p align="justify">

</p>

#### 2.11.6 Quiz
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>

#### 2.11.7 Linear regression example
<p align="justify">

</p>

#### 2.11.8 Linear regression example in JAGS
<p align="justify">

</p>

#### 2.11.9 Quiz
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>

#### 2.11.10 Reading: Applications of hierarchical modeling
<p align="justify">

</p>

#### 2.11.11 Reading: Code and data for Lesson 11
{% highlight R %}

{% endhighlight %}
<p align="justify">

</p>

#### 2.11.12 Reading: Mixture model introduction, data, and code
<p align="justify">

</p>

#### 2.11.13 Mixture model in JAGS
<p align="justify">

</p>

### 2.12 Capstone project
#### 2.12.1 Data Analysis Project
<p align="justify">

</p>

### 2.13 Honor Quiz
#### 2.13.1 Markov chains
<p align="justify">
<b>1.</b><br>
All but one of the following scenarios describes a valid Markov chain. Which one is not a Markov chain?<br>
A. Suppose you have a special savings account which accrues interest according to the following rules: the total amount deposited in a given month will earn 10(1/2) (r−1)% interest in the r-th month after the deposit. For example, if the deposits in January total 100 dollars, then you will earn 10 dollars interest in January, 5 dollars interest at the end of February, 2.50 dollars in March, etc. In addition to the interest from January, if you deposit 80 dollars in February, you will earn an additional 8 dollars at the end of February, 4 dollars at the end of March, and so forth. The total amount of money deposited in a given month follows a gamma distribution. Let $X_{t}$ be the total dollars in your account, including all deposits and interest up to the end of month t.<br>
B. At any given hour, the number of customers entering a grocery store follows a Poisson distribution. The number of customers in the store who leave during that hour also follows a Poisson distribution (only up to as many people are in the store). A clerk reports the total number of customers in the store $X_{t}$ at the end of hour t.<br>
C. Three friends take turns playing chess with the following rules: the player who sits out the current round plays the winner in the next round. Player A, who has 0.7 probability of winning any game regardless of opponent, keeps track of whether he plays in game t with an indicator variable $X_{t}$.<br>
D. While driving through a city with square blocks, you roll a six-sided die each time you come to an intersection. If the die shows 1, 2, 3, or 4, then you turn left. If the die shows 5 or 6, you turn right. Each time you reach an intersection, you report your coordinates $X_{t}$.<br><br>

<b>Answer</b>: A.<br><br>

<b>2.</b><br>
Which of the following gives the transition probability matrix for the chess example in the previous question? The first row and column correspond to X=0 (player A not playing) while the second row and column correspond to X=1 (player A playing).<br>
$$
\begin{aligned}
& \text{A.} \quad
\begin{pmatrix}
0.7 & 0 \\
0.3 & 1
\end{pmatrix} \\
\\
& \text{B.} \quad
\begin{pmatrix}
0.3 & 0 \\
0.7 & 1
\end{pmatrix} \\
\\
& \text{C.} \quad
\begin{pmatrix}
0 & 0.3 \\
1 & 0.7
\end{pmatrix} \\
\\
& \text{D.} \quad
\begin{pmatrix}
0 & 1 \\
0.3 & 0.7
\end{pmatrix} \\
\end{aligned}
$$

<b>Answer</b>: D.<br><br>

<b>3.</b><br>
Continuing the chess example, suppose that the first game is between Players B and C. What is the probability that Player A will play in Game 4? Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.79.<br>
$$\sum_{i, j = 1}^{2} P(X_{t+3} = 2 \mid X_{t+2} = i) \cdot P(X_{t+2} = i \mid P(t+1) = j) \cdot P(X_{t+1} = j \mid X_{t} = 1) = Q^{3}[1, 2]$$

<b>4.</b><br>
Which of the following is the stationary distribution for X in the chess example?<br>
A. ( .231, .769 )<br>
B. ( .750, .250 )<br>
C. ( .250, .750 )<br>
D. ( 0.0, 1.0 )<br>
E. ( .769, .231 )<br><br>

<b>Answer</b>: A.<br><br>

<b>5.</b><br>
If the players draw from the stationary distribution in Question 4 to decide whether Player A participates in Game 1, what is the probability that Player A will participate in Game 4? Round your answer to two decimal places.<br><br>

<b>Answer</b>: 0.77.<br>
This is just the stationary probability of Player A playing. If the chain starts in the stationary distribution, the probability of Player A playing in the next game, the game after that, and so forth, is always this stationary probability.<br><br>
</p>

#### 2.13.2 MCMC
<p align="justify">
<b>1.</b><br>
For Questions 1 through 3, consider the following model for data that take on values between 0 and 1:
$$
\begin{aligned}
& x_{i} \mid \alpha, \beta \stackrel{\text{i.i.d}}{\sim} \text{Beta}(\alpha, \beta), \quad i = 1, ..., n \\
& \alpha \sim \text{Gamma}(a, b) \\
& \beta \sim \text{Gamma}(r, s)
\end{aligned}
$$
where α and β are independent a priori. Which of the following gives the full conditional density for α up to proportionality?
$$
\begin{aligned}
\text{A.} \quad & P(\alpha \mid \beta, x) \propto \frac{\Gamma(\alpha + \beta)^{n}}{\Gamma(\alpha)^{n}} [\prod_{i=1}^{n} x_{i}]^{\alpha - 1} \alpha^{a-1} e^{-b \alpha} \mathbf{I}_{0 < \alpha < 1} \\
\text{B.} \quad & P(\alpha \mid \beta, x) \propto \frac{\Gamma(\alpha + \beta)^{n}}{\Gamma(\alpha)^{n} \Gamma(\beta)^{n}} [\prod_{i=1}^{n} x_{i}]^{\alpha - 1} [\prod_{i=1}^{n} (1 - x_{i})]^{\beta - 1} \alpha^{a-1} e^{-b \alpha} \beta^{r-1} e^{-s\beta} \mathbf{I}_{0 < \alpha < 1} \mathbf{I}_{0 < \beta < 1} \\
\text{C.} \quad & P(\alpha \mid \beta, x) \propto [\prod_{i=1}^{n} x_{i}]^{\alpha - 1} \alpha^{a-1} e^{-b \alpha} \mathbf{I}_{\alpha > 0} \\
\text{D.} \quad & P(\alpha \mid \beta, x) \propto \frac{\Gamma(\alpha + \beta)^{n}}{\Gamma(\alpha)^{n}} [\prod_{i=1}^{n} x_{i}]^{\alpha - 1} \alpha^{a-1} e^{-b \alpha} \mathbf{I}_{\alpha > 0}
\end{aligned}
$$
<b>Answer</b>: D.<br><br>

<b>2.</b><br>
Suppose we want posterior samples for α from the model in Question 1. What is our best option?<br>
A. The full conditional for α is not proportional to any common probability distribution, and the marginal posterior for β is not any easier, so we will have to resort to a Metropolis-Hastings sampler.<br>
B. The joint posterior for α and β is a common probability distribution which we can sample directly. Thus we can draw Monte Carlo samples for both parameters and keep the samples for α.<br>
C. The full conditional for α is proportional to a common distribution which we can sample directly, so we can draw from that.<br>
D. The full conditional for α is not a proper distribution (it doesn't integrate to 1), so we cannot sample from it.<br>
<b>Answer</b>: A.<br><br>

<b>3.</b><br>
If we elect to use a Metropolis-Hastings algorithm to draw posterior samples for α, the Metropolis-Hastings candidate acceptance ratio is computed using the full conditional for α as
$$\frac{\Gamma(\alpha)^{n} \Gamma(\alpha^{*} + \beta)^{n} [\prod_{i=1}^{n} x_{i}]^{\alpha^{*}} (\alpha^{*})^{a-1} e^{-b \alpha^{*}} q(\alpha^{*} \mid \alpha) \mathbf{I}_{\alpha^{*}>0}}{\Gamma(\alpha^{*})^{n} \Gamma(\alpha + \beta)^{n} [\prod_{i=1}^{n} x_{i}]^{\alpha} \alpha^{a-1} e^{-b \alpha} q(\alpha \mid \alpha^{*}) \mathbf{I}_{\alpha > 0}}$$
where $\alpha^{*}$ is a candidate value drawn from proposal distribution $q(\alpha^{*} \mid \alpha)$. Suppose that instead of the full conditional for α, we use the full joint posterior distribution of α and β and simply plug in the current (or known) value of β. What is the Metropolis-Hastings ratio in this case?
$$
\begin{aligned}
\text{A.} \quad & \frac{\Gamma(\alpha)^{n} \Gamma(\alpha^{*} + \beta)^{n} [\prod_{i=1}^{n} x_{i}]^{\alpha^{*}} q(\alpha^{*} \mid \alpha) \mathbf{I}_{\alpha^{*}>0}}{\Gamma(\alpha^{*})^{n} \Gamma(\alpha + \beta)^{n} [\prod_{i=1}^{n} x_{i}]^{\alpha} q(\alpha \mid \alpha^{*}) \mathbf{I}_{\alpha > 0}} \\
\text{B.} \quad & \frac{\Gamma(\alpha)^{n} \Gamma(\alpha^{*} + \beta)^{n} [\prod_{i=1}^{n} x_{i}]^{\alpha^{*}} (\alpha^{*})^{a-1} e^{-b \alpha^{*}} q(\alpha^{*} \mid \alpha) \mathbf{I}_{\alpha^{*}>0}}{\Gamma(\alpha^{*})^{n} \Gamma(\alpha + \beta)^{n} [\prod_{i=1}^{n} x_{i}]^{\alpha} \alpha^{a-1} e^{-b \alpha} q(\alpha \mid \alpha^{*}) \mathbf{I}_{\alpha > 0}} \\
\text{C.} \quad & \frac{\gamma(\alpha^{*} + \beta)^{n} [\prod_{i=1}^{n} x_{i}]^{\alpha^{*}-1} [\prod_{i=1}^{n} (1-x_{i})]^{\beta-1} (\alpha^{*})^{a-1} e^{-b \alpha^{*}} \beta^{r-1} e^{-s \beta} q(\alpha \mid \alpha^{*})\mathbf{I}_{\alpha^{*} >0} \mathbf{I}_{\beta > 0}}{\Gamma(\alpha^{*})^{n} \Gamma(\beta)^{n} q(\alpha^{*} \mid \alpha)}\\
\text{D.} \quad & \frac{(\alpha^{*})^{a-1} e^{-b \alpha^{*}} q(\alpha^{*} \mid \alpha) \mathbf{I}_{\alpha^{*}>0}}{\alpha^{a-1} e^{-b \alpha} q(\alpha \mid \alpha^{*}) \mathbf{I}_{\alpha > 0}}
\end{aligned} 
$$
<b>Answer</b>: B.<br><br>

<b>4.</b><br>
For Questions 4 and 5, re-run the Metropolis-Hastings algorithm from Lesson 4 to draw posterior samples from the model for mean company personnel growth for six new companies: (-0.2, -1.5, -5.3, 0.3, -0.8, -2.2). Use the same prior as in the lesson. Below are four possible values for the standard deviation of the normal proposal distribution in the algorithm. Which one yields the best sampling results?<br>
A. 0.5<br>
B. 1.5<br>
C. 3.0<br>
D. 4.0<br>
<b>Answer</b>: B.<br>
The candidate acceptance rate for this proposal distribution is about 0.3 which yields good results.<br><br>

<b>5.</b><br>
Report the posterior mean point estimate for μ, the mean growth, using these six data points. Round your answer to two decimal places.<br>
<b>Answer</b>: -1.48.<br><br>
</p>

#### 2.13.3 Common models and multiple factor ANOVA
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>

#### 2.13.4 Predictive distributions and mixture models
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>
