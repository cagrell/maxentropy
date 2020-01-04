def main():
    import sys, os
    import numpy as np
    import scipy as sp

    # Path to pyre fork
    # can be downloaded at https://github.com/cagrell/pyre
    dir_maxentropy_fork = 'C:\\Data\\git repos\\maxentropy\\'
    sys.path.insert(0, dir_maxentropy_fork) # The custom directory will be used instead of original pyre
    import maxentropy
    from maxentropy.utils import auxiliary_sampler_scipy
    from maxentropy.utils import evaluate_feature_matrix

    import matplotlib.pyplot as plt
    #%matplotlib inline
    plt.style.use('seaborn-darkgrid')

    # Function to invert
    def y_fun(x):
        return x*x + 1

    # True input dist
    x_true = sp.stats.norm(0.5, 0.2)

    # Generate some data 
    num_samples = 10
    y_obs = np.array([y_fun(x) for x in x_true.rvs(size = num_samples)]) 

    # Expectation functions 
    def f0(x):
        #print('evaluating f0')
        return y_fun(x)

    def f1(x):
        #print('evaluating f1')
        return y_fun(x)**2

    features = [f0, f1]
    target_expectations = [y_obs.mean(), (y_obs**2).mean()] 
    
    sampler = auxiliary_sampler_scipy(sp.stats.uniform(-1, 3), n = 100)
    #sample_xs, log_q_xs = sampler()
    #plt.hist(sample_xs, density=True)
    #plt.show()


    # Create a model
    model = maxentropy.skmaxent.MCMinDivergenceModel(None, None,
                                                 vectorized=False)

    ### Compute feature matrix manually ###
    sample_xs, log_q_xs = sampler()
    sample_F = evaluate_feature_matrix(features, sample_xs)

    model.set_sample(sample_xs, log_q_xs, sample_F)
    #model.sample = sample_xs
    #model.sample_F = log_q_xs
    #model.sample_log_probs = sample_F

    X = np.reshape(target_expectations, (1, -1))
    print('target expectations:', target_expectations)
    print('before fitting:', model.expectations())

    model.fit(X)

    print('after fitting:', model.expectations())
    print('close:', np.allclose(model.expectations(), X, atol=1e-7))

    # Plot the pdf:

    # Create pdf as a function of the feature vector
    theta = model.params
    log_Z = model.log_norm_constant()

    def log_pdf_fx(fx):
        if len(fx.shape) == 1:
            log_pdf = np.dot(theta, fx) - log_Z
        else:
            log_pdf = fx.T.dot(theta) - log_Z

        return log_pdf

    def pdf_fx(fx):
        return np.exp(log_pdf_fx(fx))

    # Set x-axis and compute features
    x = np.linspace(-1, 3, num=100)
    fx = np.array([np.array([f0(xx), f1(xx)]) for xx in x])
    pdf = np.array([pdf_fx(f) for f in fx])

    #x = np.linspace(-1, 3, num=1000)
    #pdf = model.pdf(model.features(x))
    plt.plot(x, pdf)
    plt.ylim(0, pdf.max()*1.1)
    plt.show()

if __name__== "__main__":
  main()