from __future__ import division
import numpy as np
import pandas as pd
import time
from ms_func_lib import ms_utils
import scipy.optimize as so

###############################################################
## Functions for evaluating fitting data
###############################################################
def calc_SSres(y, model_predicted_y):

    """Insert Description"""

    SSres = np.sum(np.square(y - model_predicted_y))
    return SSres


def calc_SStot(y):

    """Insert Description"""

    SStot = np.sum(np.square((y - np.mean(y))))
    return SStot


def rsq_quality_of_fit(y, model_predicted_y):

    """Insert Description"""

    SStot = calc_SStot(y)
    SSres = calc_SSres(y, model_predicted_y)
    rsq = 1 - (SSres / SStot)
    return rsq


def max_log_likelihood(x):

    """Insert Description"""

    # N = number of residuals from least squares fit
    # x = residuals from least squares fit
    N = len(x)
    return 0.5 * (-N * (np.log(2 * np.pi) + 1 - np.log(N) + np.log(np.sum(x ** 2))))


def AIC_qualityoffit(p, x):

    """Insert Description"""

    # p = number_of_fit_parameters
    # x = residuals from least squares fit
    mll = max_log_likelihood(x)
    AIC = (2 * p) - (2 * mll)
    return AIC


def AICc_qualityoffit(p, x, n):

    """Insert Description"""

    # AICc = AIC + 2p(p-1)/(n-p-1)
    # p = number_of_fit_parameters
    # x = residuals_from_fit
    # n = sample_size
    if (n - p - 1) == 0:
        AICc = 0
    else:
        AIC = AIC_qualityoffit(p, x)
        AICc = AIC + (2 * p * (p - 1) / (n - p - 1))
    return AICc


def residualVariance(y, model_predicted_y, n, p):

    """Insert Description"""

    # p = number_of_fit_parameters
    # n = sample_size
    # y = actual y values from data
    # model_predicted_y = model predicted y values from data x

    SSres = calc_SSres(y, model_predicted_y)
    resVar = SSres / (n - p)
    return resVar


###############################################################
## Generalized functions for fitting data and returning parameters
###############################################################
def model_linear(conc_peptide, slope, c):

    """Insert Description"""

    # given x (conc) and parameters, return y
    # based on
    # Fobs = slope*conc_peptide+c
    return (slope * conc_peptide) + c


def model_firstorder(conc_peptide, Fmax, Kd):

    """Insert Description"""

    # given x (conc) and parameters, return y
    # based on
    # Fobs = (Fmax*conc_peptide)/(Kd*conc_peptide)
    return (Fmax * conc_peptide) / (Kd + conc_peptide)


def model_firstorder_offset(conc_peptide, Fmax, Kd, c):

    """Insert Description"""

    # given x (conc) and parameters, return y
    # based on
    # Fobs = (Fmax*conc_peptide)/(Kd*conc_peptide)
    return (Fmax * conc_peptide) / (Kd + conc_peptide) + c


###############################################################
## Application to FP data
###############################################################
def fit_and_eval_model(function, datax, datay, p0=None):

    """Insert Description"""

    datax = datax[~np.isnan(datay)]
    datay = datay[~np.isnan(datay)]

    if p0 is None:
        if function is model_linear:
            p0 = np.asarray([20, 300])
        if function is model_firstorder:
            p0 = np.asarray([max(datay), max(datax) / 2])
        if function is model_firstorder_offset:
            p0 = np.asarray([max(datay), max(datax) / 2, 300])

    errfunc = lambda p, x, y: function(x, *p) - y

    if function is model_linear:
        ## slope, c
        fit_results = so.least_squares(errfunc, p0, method='trf', loss='soft_l1', f_scale=0.1, args=(datax, datay),
                                       bounds=([0, 0], [np.inf, np.inf]))
    elif function is model_firstorder_offset:
        ## Fmax, Kd, c
        fit_results = so.least_squares(errfunc, p0, args=(datax, datay), bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    elif function is model_firstorder:
        ## Fmax, Kd, c
        fit_results = so.least_squares(errfunc, p0, args=(datax, datay), bounds=([-100, -100], [np.inf, np.inf]))
    else:
        fit_results = None # or throw and exception here

    if function is model_linear:
        number_of_parameters = 2
    elif function is model_firstorder_offset:
        number_of_parameters = 3
    elif function is model_firstorder:
        number_of_parameters = 2
    else:
        number_of_parameters = None # or throw an exception here

    sample_size = len(datax)
    residuals = fit_results.fun

    model_predicted_y = function(datax, *fit_results.x)
    # poor measure of quality of fit
    rsq = rsq_quality_of_fit(datay, model_predicted_y)
    # better measures of quality of fit
    AICc = AICc_qualityoffit(number_of_parameters, residuals, sample_size)
    resVar = residualVariance(datay, model_predicted_y, sample_size, number_of_parameters)

    return tuple(fit_results.x) + tuple([rsq, AICc, resVar]) + (str(fit_results.success),)


def fit_models_to_data(df):

    """Insert Description"""

    ## Fit to models, evaluate and select correct fits
    start_time = time.time()
    ms_utils.print_flush('Fitting models')

    fitting_data = df.copy()

    ms_utils.print_flush('\tFitting linear model')
    ## calculations fitting a constant offset
    (
        fitting_data['linear_slope'],
        fitting_data['linear_offset'],
        fitting_data['linear_rsq'],
        fitting_data['linear_AICc'],
        fitting_data['linear_resVar'],
        fitting_data['linear_success']
    ) = np.vectorize(fit_and_eval_model)(model_linear, fitting_data['PeptideConc'], fitting_data['Fluorescence'])
    ms_utils.print_flush('\tLinear models complete. Time elapsed: ',time.time()-start_time, 'seconds.')

    ms_utils.print_flush('\tFitting first-order model')
    (
        fitting_data['fo_Fmax'],
        fitting_data['fo_Kd'],
        fitting_data['fo_offset'],
        fitting_data['fo_rsq'],
        fitting_data['fo_AICc'],
        fitting_data['fo_resVar'],
        fitting_data['fo_success']
    ) = np.vectorize(fit_and_eval_model)(model_firstorder_offset, fitting_data['PeptideConc'],
                                         fitting_data['Fluorescence'])
    ms_utils.print_flush('\tFirst-order complete. Time elapsed: ', time.time() - start_time, 'seconds.')

    ms_utils.print_flush('Fitting Complete', time.time() - start_time, 'elapsed')

    return fitting_data


###############################################################
## Model selection
###############################################################

def evaluate_fits(df):
    start_time = time.time()
    ms_utils.print_flush('Evaluating Fits')

    # Initialize lists for storing for model selection
    fit_method_list = []
    fit_Kd_list = []
    fit_Fmax_list = []
    fit_rsq_list = []
    fit_AICc_list = []
    fit_resVar_list = []
    fit_success_list = []
    fit_offset_list = []
    fit_slope_list = []
    fit_residuals_list = []
    fit_snr_list = []
    fit_dnr_list = []
    fit_sumres_list = []
    fit_modelDNR_list= []
    fit_bindingcall_list = []


    for row in df.itertuples():

        fl_vals = row.Fluorescence
        conc_vals = row.PeptideConc
        y_hat_linear = [model_linear(x, row.linear_slope, row.linear_offset) for x in conc_vals]
        y_hat_fo = [model_firstorder_offset(x, row.fo_Fmax, row.fo_Kd, row.fo_offset) for x in conc_vals]
        max_x = max([x for x in conc_vals])

        linear_better_than_fo = row.linear_AICc <= row.fo_AICc
        fo_better_than_linear = row.fo_AICc <= row.linear_AICc

        # linear model chosen if:
        #           linear is best
        #           firstorder model is best, but there are fo artifacts

        # firstorder model chosen if:
        #           firstorder is best, and the first order fit is not an artifact

        # initialize values
        linear_best = False
        fo_best = False

        # define artifacts
        fo_artifacts = (row.fo_Kd > 1000 or row.fo_Fmax < 1 or row.fo_offset < 100)

        # rules encoded
        if linear_better_than_fo or (fo_better_than_linear and fo_artifacts):
                linear_best = True
        elif fo_better_than_linear and ~fo_artifacts:
            fo_best = True
        else:
            ms_utils.print_flush('Model classification error')
            break



        #
        # Evauate and store fits
        #
        if linear_best:
            fit_method_list.append('linear')
            fit_Kd_list.append(None)
            fit_Fmax_list.append(None)
            fit_slope_list.append(row.linear_slope)
            fit_rsq_list.append(row.linear_rsq)
            fit_AICc_list.append(row.linear_AICc)
            fit_resVar_list.append(row.linear_resVar)
            fit_success_list.append(row.linear_success)
            fit_offset_list.append(row.linear_offset)
            y_hat = y_hat_linear
            resid = fl_vals - y_hat

            fit_residuals_list.append(resid)

            # if fit is linear, then make binding calls based on slope
            # slope <= 5 is non-binder
            # slope > 5 is aggregator
            if row.linear_slope <= 5:
                fit_bindingcall_list.append('nobinding')
            else:
                fit_bindingcall_list.append('aggregator')

        elif fo_best:

            fit_method_list.append('fo')
            fit_Kd_list.append(row.fo_Kd)
            fit_Fmax_list.append(row.fo_Fmax)
            fit_slope_list.append(None)
            fit_rsq_list.append(row.fo_rsq)
            fit_AICc_list.append(row.fo_AICc)
            fit_resVar_list.append(row.fo_resVar)
            fit_success_list.append(row.fo_success)
            fit_offset_list.append(row.fo_offset)
            y_hat = y_hat_fo
            resid = fl_vals - y_hat

            fit_residuals_list.append(resid)

            # if fit is base model (one-to-one kinetics)
            # then filter out known bad fits

            # snr < 1 should be held out as 'low-SNR' to review,
            # it must be done after the loop, so here call it a binder

            fit_bindingcall_list.append('binder')
        else:
            fit_bindingcall_list.append('classification error')
            y_hat = 0
            resid = 0


        fit_modelDNR = abs(max(y_hat) - min(y_hat))
        fit_sum_resid = np.sum(abs(resid))
        fit_snr = fit_modelDNR / fit_sum_resid
        fit_dnr = abs(fl_vals.max() - fl_vals.min())

        fit_modelDNR_list.append(fit_modelDNR)
        fit_sumres_list.append(fit_sum_resid)
        fit_snr_list.append(fit_snr)
        fit_dnr_list.append(fit_dnr)

    df['fit_method'] = pd.Series(fit_method_list, index=df.index)
    df['fit_slope'] = pd.Series(fit_slope_list, index=df.index)
    df['fit_Kd'] = pd.Series(fit_Kd_list, index=df.index)
    df['fit_Fmax'] = pd.Series(fit_Fmax_list, index=df.index)
    df['fit_offset'] = pd.Series(fit_offset_list, index=df.index)
    df['fit_rsq'] = pd.Series(fit_rsq_list, index=df.index)
    df['fit_AICc'] = pd.Series(fit_AICc_list, index=df.index)
    df['fit_resVar'] = pd.Series(fit_resVar_list, index=df.index)
    df['fit_success'] = pd.Series(fit_success_list, index=df.index)
    df['fit_residuals'] = pd.Series(fit_residuals_list, index=df.index)
    df['fit_modelDNR'] = pd.Series(fit_modelDNR_list, index=df.index)
    df['fit_sumres'] = pd.Series(fit_sumres_list, index=df.index)
    df['fit_snr'] = pd.Series(fit_snr_list, index=df.index)
    df['fit_dnr'] = pd.Series(fit_dnr_list, index=df.index)
    df['binding_call'] = pd.Series(fit_bindingcall_list, index=df.index)

    # mark some binders as 'low-SNR' until those cases are resolved
    df.loc[(df.binding_call == 'binder') & (df.fit_snr < 1), 'binding_call'] = 'low-SNR'

    ms_utils.print_flush('\tEvaluation Complete', time.time() - start_time, 'elapsed')
    return df


