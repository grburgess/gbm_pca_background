import numpy as np
import warnings
import scipy.optimize as opt
import numdifftools as nd


class Polynomial(object):
    def __init__(self, coefficients, is_integral=False):
        """

        :param coefficients: array of poly coefficients
        :param is_integral: if this polynomial is an
        """
        self._coefficients = coefficients
        self._degree = len(coefficients) - 1

        self._i_plus_1 = np.array(range(1, self._degree + 1 + 1), dtype=float)

        self._cov_matrix = np.zeros((self._degree + 1, self._degree + 1))

        # we can fix some things for speed
        # we only need to set the coeff for the
        # integral polynomial
        if not is_integral:

            integral_coeff = [0]

            integral_coeff.extend(
                map(lambda i: self._coefficients[i - 1] / float(i),
                    range(1, self._degree + 1 + 1)))

            self._integral_polynomial = Polynomial(
                integral_coeff, is_integral=True)

    @property
    def degree(self):
        """
        the polynomial degree
        :return:
        """
        return self._degree

    @property
    def error(self):
        """
        the error on the polynomial coefficients
        :return:
        """
        return np.sqrt(self._cov_matrix.diagonal())

    def __get_coefficient(self):
        """ gets the coefficients"""

        return self._coefficients

    def ___get_coefficient(self):
        """ Indirect coefficient getter """

        return self.__get_coefficient()

    def __set_coefficient(self, val):
        """ sets the coefficients"""

        self._coefficients = val

        integral_coeff = [0]

        integral_coeff.extend(
            map(lambda i: self._coefficients[i - 1] / float(i),
                range(1, self._degree + 1 + 1)))

        self._integral_polynomial = Polynomial(
            integral_coeff, is_integral=True)

    def ___set_coefficient(self, val):
        """ Indirect coefficient setter """

        return self.__set_coefficient(val)

    coefficients = property(
        ___get_coefficient,
        ___set_coefficient,
        doc="""Gets or sets the coefficients of the polynomial.""")

    def __call__(self, x):

        result = 0
        for coefficient in self._coefficients[::-1]:
            result = result * x + coefficient
        return result

    def compute_covariance_matrix(self, function, best_fit_parameters):
        """
        Compute the covariance matrix of this fit
        :param function: the loglike for the fit
        :param best_fit_parameters: the best fit parameters
        :return:
        """

        minima = np.zeros_like(best_fit_parameters) - 100
        maxima = np.zeros_like(best_fit_parameters) + 100

        try:

            hessian_matrix = get_hessian(function, best_fit_parameters, minima,
                                         maxima)

        except ParameterOnBoundary:

            warnings.warn(
                "One or more of the parameters are at their boundaries. Cannot compute covariance and"
                " errors", CannotComputeCovariance)

            n_dim = len(best_fit_parameters)

            self._cov_matrix = np.zeros((n_dim, n_dim)) * np.nan

        # Invert it to get the covariance matrix

        try:

            covariance_matrix = np.linalg.inv(hessian_matrix)

            self._cov_matrix = covariance_matrix

        except:

            warnings.warn(
                "Cannot invert Hessian matrix, looks like the matrix is singluar"
            )

            n_dim = len(best_fit_parameters)

            self._cov_matrix = np.zeros((n_dim, n_dim)) * np.nan

    @property
    def covariance_matrix(self):
        return self._cov_matrix

    def integral(self, xmin, xmax):
        """ 
        Evaluate the integral of the polynomial between xmin and xmax

        """

        return self._integral_polynomial(xmax) - self._integral_polynomial(
            xmin)

    def _eval_basis(self, x):

        return (1. / self._i_plus_1) * np.power(x, self._i_plus_1)

    def integral_error(self, xmin, xmax):
        """
        computes the integral error of an interval
        :param xmin: start of the interval
        :param xmax: stop of the interval
        :return: interval error
        """
        c = self._eval_basis(xmax) - self._eval_basis(xmin)
        tmp = c.dot(self._cov_matrix)
        err2 = tmp.dot(c)

        return np.sqrt(err2)


class PolyLogLikelihood(object):
    """
    Implements a Poisson likelihood (i.e., the Cash statistic). Mind that this is not
    the Castor statistic (Cstat). The difference between the two is a constant given
    a dataset. I kept Cash instead of Castor to make easier the comparison with ROOT
    during tests, since ROOT implements the Cash statistic.
    """

    def __init__(self, x, y, model, exposure):
        self._bin_centers = x
        self._counts = y
        self._model = model
        self._parameters = model.coefficients
        self._exposure = exposure

        def cov_call(*parameters):
            self._model.coefficients = parameters
            M = self._model(self._bin_centers) * self._exposure
            M_fixed, tiny = self._fix_precision(M)

            # Replace negative values for the model (impossible in the Poisson context)
            # with zero

            negative_mask = (M < 0)
            if (len(negative_mask.nonzero()[0]) > 0):
                M[negative_mask] = 0.0

            # Poisson loglikelihood statistic (Cash) is:
            # L = Sum ( M_i - D_i * log(M_i))

            logM = self._evaluate_logM(M)

            # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
            # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
            # to zero, then overwrite the elements corresponding to D_i > 0

            d_times_logM = np.zeros(len(self._counts))

            non_zero_mask = (self._counts > 0)

            d_times_logM[non_zero_mask] = self._counts[non_zero_mask] * logM[
                non_zero_mask]

            log_likelihood = np.sum(M_fixed - d_times_logM)

            return log_likelihood

        self.cov_call = cov_call

    def _evaluate_logM(self, M):
        # Evaluate the logarithm with protection for negative or small
        # numbers, using a smooth linear extrapolation (better than just a sharp
        # cutoff)
        tiny = np.float64(np.finfo(M[0]).tiny)

        non_tiny_mask = (M > 2.0 * tiny)

        tink_mask = np.logical_not(non_tiny_mask)

        if (len(tink_mask.nonzero()[0]) > 0):
            logM = np.zeros(len(M))
            logM[tink_mask] = np.abs(M[tink_mask]) / tiny + np.log(tiny) - 1
            logM[non_tiny_mask] = np.log(M[non_tiny_mask])

        else:

            logM = np.log(M)

        return logM

    def __call__(self, parameters):
        """
          Evaluate the Cash statistic for the given set of parameters
        """

        # Compute the values for the model given this set of parameters
        # model is in counts

        self._model.coefficients = parameters
        M = self._model(self._bin_centers) * self._exposure
        M_fixed, tiny = self._fix_precision(M)

        # Replace negative values for the model (impossible in the Poisson context)
        # with zero

        negative_mask = (M < 0)
        if (len(negative_mask.nonzero()[0]) > 0):
            M[negative_mask] = 0.0

        # Poisson loglikelihood statistic (Cash) is:
        # L = Sum ( M_i - D_i * log(M_i))

        logM = self._evaluate_logM(M)

        # Evaluate v_i = D_i * log(M_i): if D_i = 0 then the product is zero
        # whatever value has log(M_i). Thus, initialize the whole vector v = {v_i}
        # to zero, then overwrite the elements corresponding to D_i > 0

        d_times_logM = np.zeros(len(self._counts))

        non_zero_mask = (self._counts > 0)

        d_times_logM[non_zero_mask] = self._counts[non_zero_mask] * logM[
            non_zero_mask]

        log_likelihood = np.sum(M_fixed - d_times_logM)

        return log_likelihood

    def _fix_precision(self, v):
        """
          Round extremely small number inside v to the smallest usable
          number of the type corresponding to v. This is to avoid warnings
          and errors like underflows or overflows in math operations.
        """
        tiny = np.float64(np.finfo(v[0]).tiny)
        zero_mask = (np.abs(v) <= tiny)
        if (len(zero_mask.nonzero()[0]) > 0):
            v[zero_mask] = np.sign(v[zero_mask]) * tiny

        return v, tiny


def polyfit(x, y, grade, exposure):
    """ function to fit a polynomial to event data. not a member to allow parallel computation """

    # Check that we have enough counts to perform the fit, otherwise
    # return a "zero polynomial"
    non_zero_mask = (y > 0)
    n_non_zero = len(non_zero_mask.nonzero()[0])
    if n_non_zero == 0:
        # No data, nothing to do!
        return Polynomial([0.0]), 0.0

    # Compute an initial guess for the polynomial parameters,
    # with a least-square fit (with weight=1) using SVD (extremely robust):
    # (note that polyfit returns the coefficient starting from the maximum grade,
    # thus we need to reverse the order)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        initial_guess = np.polyfit(x, y, grade)

    initial_guess = initial_guess[::-1]

    polynomial = Polynomial(initial_guess)

    # Check that the solution found is meaningful (i.e., definite positive
    # in the interval of interest)
    M = polynomial(x)

    negative_mask = (M < 0)

    if len(negative_mask.nonzero()[0]) > 0:
        # Least square fit failed to converge to a meaningful solution
        # Reset the initialGuess to reasonable value
        initial_guess[0] = np.mean(y)
        meanx = np.mean(x)
        initial_guess = map(lambda x: abs(x[1]) / pow(meanx, x[0]),
                            enumerate(initial_guess))

    # Improve the solution using a logLikelihood statistic (Cash statistic)
    log_likelihood = PolyLogLikelihood(x, y, polynomial, exposure)

    # Check that we have enough non-empty bins to fit this grade of polynomial,
    # otherwise lower the grade
    dof = n_non_zero - (grade + 1)

    if dof <= 2:
        # Fit is poorly or ill-conditioned, have to reduce the number of parameters
        while (dof < 2 and len(initial_guess) > 1):
            initial_guess = initial_guess[:-1]
            polynomial = Polynomial(initial_guess)
            log_likelihood = PolyLogLikelihood(x, y, polynomial, exposure)

    # Try to improve the fit with the log-likelihood


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        final_estimate = opt.minimize(log_likelihood, initial_guess)['x']
        final_estimate = np.atleast_1d(final_estimate)

    # Get the value for cstat at the minimum

    min_log_likelihood = log_likelihood(final_estimate)

    # Update the polynomial with the fitted parameters,
    # and the relative covariance matrix

    final_polynomial = Polynomial(final_estimate)

    final_polynomial.compute_covariance_matrix(log_likelihood.cov_call,
                                               final_estimate)

    return final_polynomial, min_log_likelihood


import numpy as np


class ParameterOnBoundary(RuntimeError):
    pass


class CannotComputeHessian(RuntimeError):
    pass


def _get_wrapper(function, point, minima, maxima):

    point = np.array(point, ndmin=1, dtype=float)
    minima = np.array(minima, ndmin=1, dtype=float)
    maxima = np.array(maxima, ndmin=1, dtype=float)

    n_dim = point.shape[0]

    orders_of_magnitude = 10**np.ceil(
        np.log10(np.abs(point)))  # type: np.ndarray

    scaled_point = point / orders_of_magnitude
    scaled_minima = minima / orders_of_magnitude
    scaled_maxima = maxima / orders_of_magnitude

    # Decide a delta for the finite differentiation
    # The algorithm implemented in numdifftools is robust with respect to the choice
    # of delta, as long as we are not going beyond the boundaries (which would cause
    # the procedure to fail)

    scaled_deltas = np.zeros_like(scaled_point)

    for i in range(n_dim):

        scaled_value = scaled_point[i]

        scaled_min_value, scaled_max_value = (scaled_minima[i],
                                              scaled_maxima[i])

        if scaled_value == scaled_min_value or scaled_value == scaled_max_value:

            raise ParameterOnBoundary(
                "Value for parameter number %s is on the boundary" % i)

        if not np.isnan(scaled_min_value):

            # Parameter with low bound

            distance_to_min = scaled_value - scaled_min_value

        else:

            # No defined minimum

            distance_to_min = np.inf

        if not np.isnan(scaled_max_value):

            # Parameter with hi bound

            distance_to_max = scaled_max_value - scaled_value

        else:

            # No defined maximum

            distance_to_max = np.inf

        # Delta is the minimum between 3% of the value, and 1/2.5 times the minimum
        # distance to either boundary. 1/2 of that factor is due to the fact that numdifftools uses
        # twice the delta to compute the differential, and the 0.5 is due to the fact that we don't want
        # to go exactly equal to the boundary

        scaled_deltas[i] = min([
            0.03 * abs(scaled_point[i]), distance_to_max / 2.5, distance_to_min
            / 2.5
        ])

    def wrapper(x):

        scaled_back_x = x * orders_of_magnitude  # type: np.ndarray

        try:

            result = function(*scaled_back_x)

        except:

            raise CannotComputeHessian(
                "Cannot compute Hessian, parameters out of bounds at %s" %
                scaled_back_x)

        else:

            return result

    return wrapper, scaled_deltas, scaled_point, orders_of_magnitude, n_dim


def get_hessian(function, point, minima, maxima):

    wrapper, scaled_deltas, scaled_point, orders_of_magnitude, n_dim = _get_wrapper(
        function, point, minima, maxima)

    # Compute the Hessian matrix at best_fit_values

    hessian_matrix_ = nd.Hessian(wrapper, scaled_deltas)(scaled_point)

    # Transform it to numpy matrix

    hessian_matrix = np.array(hessian_matrix_)

    # Now correct back the Hessian for the scales
    for i in range(n_dim):

        for j in range(n_dim):

            hessian_matrix[i, j] /= orders_of_magnitude[
                i] * orders_of_magnitude[j]

    return hessian_matrix
