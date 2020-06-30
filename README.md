# NHPoKD
A model to mine person’s typing patterns of keystroke event sequence using a small amount of data.
In practice, the amount of keystroke data available for training model is often insufficient. In order to make full use of the temporal characteristics and reduce the size of training data, we use multivariate Hawkes processes which can discover triggering relations of events of a sequence. Considering keystroke behavior is unstable over time, we introduce disturbance term of conditional intensity function. In order to propose a model that can learn a general representation of the underlying dynamics from the event history without assuming a fixed parametric form of triggering function, we use Bayesian non-parametric inference method. We modified Salehi and Trouleau's model(https://github.com/trouleau/var-hawkes) and build NHPoKD.
# DATA
We analyis the CMU dataset. it can be obtained form link: http://www.cs.cmu.edu/~keystroke/ksk-thesis/

