import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits

import warnings

from step_plot import slice_disjoint, step_plot
from poly_fitting import polyfit
from significance import Significance

class CSPECData(object):
    def __init__(self, cspec_file, postion_history=None):
        """
        Read and process GBM daily CSPEC files

        :param cspec_file: CSPEC file name
        :param postion_history: option position history file
        """

        # extract the appropriate information

        with fits.open(cspec_file) as f:

            # spectrum extension

            spec_ext = f['SPECTRUM']

            self._exposure = spec_ext.data['EXPOSURE']
            self._start_times = spec_ext.data['TIME']
            self._stop_times = spec_ext.data['ENDTIME']
            self._counts = spec_ext.data['COUNTS']

        self._good_bkgs = []

        # create an array of total counts
        self._total_counts = self._counts.sum(axis=1)

        # get the durations
        self._widths = self._stop_times - self._start_times

        # mean times used for polynomial fitting
        self._means = np.mean(zip(self._start_times, self._stop_times), axis=1)

        # first get rid of all data drop outs
        # and divide up the intervals
        self._break_up_intervals()

        # select only intervals with no triggers
        self._filter_burst_intervals()

        # fit polynomials to all those intervals
        self._fit_all_intervals()

        # extract all the intervals that had
        # no excess and a decent fit
        self._extract_good()

        # create the spectra
        self._create_spectra()

    def _break_up_intervals(self):
        """
        breaks up non-zero intervals into sets 
        """

        # get all the intervals that have counts

        non_zero_mask = self._total_counts > 0

        # convert that to an index array
        # and get the intervals in sets

        self._intervals = slice_disjoint(non_zero_mask.nonzero()[0])

        self._n_intervals = len(self._intervals)

    def _filter_burst_intervals(self):
        """
        marks intervals that have bin sizes != 4.096
        as these are triggered intervals

        """

        self._drop_idx = []

        for i, (start, stop) in enumerate(self._intervals):

            # get the durations for this interval
            this_selection = self._widths[start:stop]

            # the intervals sizes are not exact
            # so we extract them if they are about 4.096 sec
            if not np.allclose(this_selection, 4.096, atol=1E-3):

                self._drop_idx.append(i)

    def _fit_all_intervals(self):
        """
        Fit polynomials to all intervals and save the values

        """

        self._polys = []
        self._poly_start = []
        self._poly_stop = []
        self._good = []
        self._zero_time = []

        for i, (start, stop) in enumerate(self._intervals):

            # only fit those intervals that are 'good'
            # so far

            if i not in self._drop_idx:

                self._poly_fit_interval(i)

    def _poly_fit_interval(self, interval_number, step=150):
        """
        Fitting of an interval and testing for significant deviation

        """

        # extract the start and stop indices
        start, stop = self._intervals[interval_number]

        # we do a while loop so that we only grab intervals
        # long enough be used. otherwise, we will grab bad intervals
        # as well
        ii = start

        while (ii <= stop):

            # check if this interval is not
            # running over the end of the total interval
            if ii + step <= stop:

                # extract the mean times
                xs = self._means[ii:ii + step]

                # we will normalize in time to
                # keep the numbers sane

                zero_time = np.median(xs)

                tt = xs - zero_time

                # get the exposure and total counts
                ys = self._total_counts[ii:ii + step]
                es = self._exposure[ii:ii + step]

                # try several order polynomials
                min_grade = 0
                max_grade = 5
                log_likelihoods = []

                for grade in range(min_grade, max_grade + 1):
                    polynomial, log_like = polyfit(tt, ys, grade, exposure=es)

                    log_likelihoods.append(log_like)

                # Found the best one
                delta_loglike = np.array(
                    map(lambda x: 2 * (x[0] - x[1]),
                        zip(log_likelihoods[:-1], log_likelihoods[1:])))

                delta_threshold = 9.0

                mask = (delta_loglike >= delta_threshold)

                if (len(mask.nonzero()[0]) == 0):

                    # best grade is zero!
                    best_grade = 0

                else:

                    best_grade = mask.nonzero()[0][-1] + 1

                # refit fit with the best polynomial

                poly, mll = polyfit(tt, ys, best_grade, exposure=es)

                # save all the info
                self._polys.append(poly)
                self._poly_start.append(ii)
                self._poly_stop.append(ii + step)
                self._zero_time.append(zero_time)

                # now get the significance:

                sigma_bs = []
                estimated = []

                for tstart, tstop in zip(
                        self._start_times[ii:ii + step] - zero_time,
                        self._stop_times[ii:ii + step] - zero_time):

                    sigma_bs.append(poly.integral_error(tstart, tstop))
                    estimated.append(poly.integral(tstart, tstop))

                sig = Significance(self._total_counts[ii:ii + step],
                                   np.array(estimated))
                all_sig = sig.li_and_ma_equivalent_for_gaussian_background(
                    np.array(sigma_bs))

                # if the data significantly deviate from the
                # polynomial, we discard it

                sig_threshold = 5

                if np.any(np.abs(all_sig) > sig_threshold):

                    self._good.append(False)
                else:

                    self._good.append(True)

            # iterate

            ii += step

    def _calculate_good_background(self):
        """
        Calculate all the good polynomial estimations
        """

        for p, s, e, g, zt in zip(self._polys, self._poly_start,
                                  self._poly_stop, self._good,
                                  self._zero_time):

            # if the interval is good
            if g:

                # get the background initialized
                bkg = []

                for ii, jj, kk in zip(self._start_times[s:e] - zt,
                                      self._stop_times[s:e] - zt,
                                      self._exposure[s:e]):

                    # integrate the polynomial over
                    # the intervals and convert to a rate
                    bkg.append(p.integral(ii, jj) / kk)

                # save it
                self._good_bkgs.append(bkg)

    def _extract_good_poly_intervals(self):
        """
        Extract all the intervals with no significant excess
        """
        self._good_intervals = []

        for p, s, e, g, zt in zip(self._polys, self._poly_start,
                                  self._poly_stop, self._good,
                                  self._zero_time):
            if g:

                self._good_intervals.append([s, e])

    def _create_spectra(self):
        """
        Create count spectra from the selected intervals
        """

        self._spectra = []
        self._total_exposure = []

        # the step size is 4.096*step
        # this corresponds to about 20 seconds right now

        step = 5

        for i, (start, stop) in enumerate(self._good_intervals):

            # we again do not want to include intervals
            # that are bad so we are careful to not jump over the
            # bounds of the intervals
            for j in xrange(start, stop, step):
                spectrum = np.zeros(self._counts.shape[1])
                exposure = 0.

                # make sure that we do not run over
                if j + step <= stop:

                    for k in range(step):

                        # sum up the intervals

                        exposure += self._exposure[j + k]
                        spectrum += self._counts[j + k, :]

                    self._spectra.append(spectrum)
                    self._total_exposure.append(exposure)

    def plot_all(self, show_bad=False, show_poly=False):
        """
        Plot all the the intervals
        :param show_bad: plot the good and bad intervals
        :param show_poly: show the polynomial fits


        """
        fig, ax = plt.subplots()

        if show_bad:

            # simply color the light curve differently 

            for i in xrange(self._n_intervals):

                if i in self._drop_idx:

                    color = 'r'

                else:

                    color = 'g'

                self._plot_interval(i, ax, color=color)

        else:

            # show only the intervals without triggers
            # this still includes those which are
            # rejected by the poly fit

            itr = 0
            for i, (start, stop) in enumerate(self._intervals):

                if i not in self._drop_idx:

                    self._plot_interval(i, ax)

                    itr += 1

        if show_poly:

            # show all the poly fits
            # but color good and bad intervals
            # differently
            
            for p, s, e, g, zt in zip(self._polys, self._poly_start,
                                      self._poly_stop, self._good,
                                      self._zero_time):

                time = self._means[s:e]

                bkg = []

                for ii, jj, kk in zip(self._start_times[s:e] - zt,
                                      self._stop_times[s:e] - zt,
                                      self._exposure[s:e]):

                    bkg.append(p.integral(ii, jj) / kk)

                if g:

                    color = 'w'

                else:

                    color = 'grey'

                ax.plot(time, bkg, color=color)

    def plot_good(self):
        """
        Plot only the good intervals along with the polynomial
        fits
        """

        fig, ax = plt.subplots()

        if not self._good_bkgs:

            self._calculate_good_background()
        
        for bkg, (start, stop) in zip(self._good_bkgs, self._good_intervals):

            time = self._means[start:stop]

            self._hist_slice(start, stop, ax, color='b')

            ax.plot(time, bkg, color='w', lw=0.8)

    def _hist_slice(self, start, stop, ax, **kwargs):

        bins = zip(self._start_times[start:stop], self._stop_times[start:stop])
        counts = self._total_counts[start:stop]
        exposure = self._exposure[start:stop]

        step_plot(xbins=bins, y=counts / exposure, ax=ax, **kwargs)

    def _plot_interval(self, interval_number, ax, **kwargs):

        start, stop = self._intervals[interval_number]

        self._hist_slice(start, stop, ax, **kwargs)
