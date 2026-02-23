# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:03:06 2024

@author: poilo
"""

import numpy as np
from scipy.special import gamma, polygamma

def two(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Calculate the denominator components
    gamma_half_5 = gamma(gamma_value / 2)
    gamma_half_6 = gamma(delta_value / 2)
    
    denom_1 = gamma((gamma_value + delta_value - 3) / 2) * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma((gamma_value + delta_value - 1) / 2)
    
    denom = denom_1 - denom_2
    
    # Calculate the numerator
    numerator = (gamma_value + delta_value - 2 - 2 * beta_value) * gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4
    
    # Final result
    result = numerator / (2 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    return result

def three(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Calculate the Polygamma function (derivative of Gamma)
    polygamma_term = polygamma(0, (gamma_value - 1) / 2)
    
    # Calculate the denominator components
    gamma_half_5 = gamma(gamma_value / 2)
    gamma_half_6 = gamma(delta_value / 2)
    
    denom_1 = gamma((gamma_value + delta_value - 3) / 2) * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma((gamma_value + delta_value - 1) / 2)
    
    denom = denom_1 - denom_2
    
    # Calculate the numerator
    numerator = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * polygamma_term
    
    # Final result
    result = numerator / (4 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    return result


def four(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Calculate the Polygamma function
    polygamma_term = polygamma(0, gamma_value / 2)
    
    # Calculate the denominator components
    gamma_half_5 = gamma(gamma_value / 2)
    gamma_half_6 = gamma(delta_value / 2)
    
    denom_1 = gamma((gamma_value + delta_value - 3) / 2) * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma((gamma_value + delta_value - 1) / 2)
    
    denom = denom_1 - denom_2
    
    # Calculate the numerator
    numerator = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * polygamma_term
    
    # Final result
    result = - numerator / (4 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    return result

def five(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Calculate the Polygamma function
    polygamma_term = polygamma(0, (gamma_value + delta_value - 2) / 2)
    
    # Calculate the denominator components
    gamma_half_5 = gamma(gamma_value / 2)
    gamma_half_6 = gamma(delta_value / 2)
    
    denom_1 = gamma((gamma_value + delta_value - 3) / 2) * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma((gamma_value + delta_value - 1) / 2)
    
    denom = denom_1 - denom_2
    
    # Calculate the numerator
    numerator = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * polygamma_term
    
    # Final result
    result = numerator / (4 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    return result

def six(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Calculate the Polygamma function
    polygamma_term = polygamma(0, (gamma_value + delta_value) / 2)
    
    # Calculate the denominator components
    gamma_half_5 = gamma(gamma_value / 2)
    gamma_half_6 = gamma(delta_value / 2)
    
    denom_1 = gamma((gamma_value + delta_value - 3) / 2) * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma((gamma_value + delta_value - 1) / 2)
    
    denom = denom_1 - denom_2
    
    # Calculate the numerator
    numerator = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * polygamma_term
    
    # Final result
    result = numerator / (4 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    return result

def seven(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Calculate additional Gamma terms for the complex numerator
    gamma_half_5 = gamma((gamma_value + delta_value - 3) / 2)
    gamma_half_6 = gamma((gamma_value + delta_value - 1) / 2)
    
    # Calculate the Polygamma functions
    polygamma_term_1 = polygamma(0, (gamma_value + delta_value - 3) / 2)
    polygamma_term_2 = polygamma(0, (gamma_value + delta_value - 2) / 2)
    polygamma_term_3 = polygamma(0, (gamma_value + delta_value - 1) / 2)
    polygamma_term_4 = polygamma(0, (gamma_value + delta_value) / 2)
    
    # Calculate the denominator components
    gamma_half_7 = gamma(gamma_value / 2)
    gamma_half_8 = gamma(delta_value / 2)
    
    denom_1 = gamma_half_5 * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma_half_6
    
    denom = denom_1 - denom_2
    
    # Calculate the numerator part
    part_1 = 0.5 * gamma_half_5 * gamma_half_4 * polygamma_term_1
    part_2 = -0.5 * beta_value * gamma_half_3 * gamma_half_6 * polygamma_term_2
    part_3 = -0.5 * beta_value * gamma_half_3 * gamma_half_6 * polygamma_term_3
    part_4 = 0.5 * gamma_half_5 * gamma_half_4 * polygamma_term_4
    
    # Full numerator
    numerator = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * (part_1 + part_2 + part_3 + part_4)
    
    # Final result
    result = - numerator / (2 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_7 * gamma_half_8 * denom**2)
    
    return result




def df_gamma(gamma_value, delta_value, beta_value):
    gamma_half_1 = gamma((gamma_value - 1.) / 2.)
    gamma_half_2 = gamma((delta_value - 1.) / 2.)
    gamma_half_3 = gamma((gamma_value + delta_value - 2.) / 2.)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2.)
    
    # Denominators
    gamma_half_5 = gamma(gamma_value / 2.)
    gamma_half_6 = gamma(delta_value / 2.)
    
    denom_1 = gamma_half_4 * gamma((gamma_value + delta_value - 3.) / 2.)
    denom_2 = beta_value * gamma_half_3 * gamma((gamma_value + delta_value - 1.) / 2.)
    
    denom = denom_1 - denom_2
    
    # Common terms for the derivative
    term_1 = (gamma_value + delta_value - 5.) * gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4
    term_2 = (-2. - 2. * beta_value + gamma_value + delta_value)
    
    # First part of the derivative
    first_part = (term_1) / (2. * np.sqrt(np.pi) * (delta_value - 3.) * gamma_half_5 * gamma_half_6 * denom)
    # print(first_part)
    # Second part of the derivative
    second_part = two(gamma_value, delta_value, beta_value)
    # print(second_part)
    # Third part of the derivative
    third_part = three(gamma_value, delta_value, beta_value)
    # print(third_part)
    # Fourth part of the derivative
    fourth_part = four(gamma_value, delta_value, beta_value)
    # print(fourth_part)
    # Fifth part of the derivative
    fifth_part = five(gamma_value, delta_value, beta_value)
    # print(fifth_part)
    sixth_part = six(gamma_value, delta_value, beta_value)
    # print(sixth_part)
    seventh_part = seven(gamma_value, delta_value, beta_value)
    # print(seventh_part)
    # Combine all the parts
    return first_part + second_part + third_part + fourth_part + fifth_part + sixth_part+seventh_part

# print(df_gamma(2.1,2.5,0.1))

def deltaone(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Denominator components
    gamma_half_5 = gamma(gamma_value / 2)
    gamma_half_6 = gamma(delta_value / 2)
    
    denom_1 = gamma((gamma_value + delta_value - 3) / 2) * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma((gamma_value + delta_value - 1) / 2)
    
    denom = denom_1 - denom_2
    
    # First term in the sum
    term_1 = (gamma_value + delta_value - 5) * gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4
    term_1 /= (2 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    # Second term in the sum
    term_2 = (gamma_value + delta_value - 2 - 2 * beta_value) * gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4
    term_2 /= (2 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    # Third term in the sum
    term_3 = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    term_3 *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4
    term_3 /= (2 * np.sqrt(np.pi) * (delta_value - 3)**2 * gamma_half_5 * gamma_half_6 * denom)
    
    # Final result
    result = term_1 + term_2 - term_3
    
    return result

def deltatwo(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Calculate the Polygamma functions
    polygamma_term_1 = polygamma(0, (delta_value - 1) / 2)
    polygamma_term_2 = polygamma(0, delta_value / 2)
    
    # Calculate the denominator components
    gamma_half_5 = gamma(gamma_value / 2)
    gamma_half_6 = gamma(delta_value / 2)
    
    denom_1 = gamma((gamma_value + delta_value - 3) / 2) * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma((gamma_value + delta_value - 1) / 2)
    
    denom = denom_1 - denom_2
    
    # Calculate the numerator part for the first term
    numerator_1 = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator_1 *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * polygamma_term_1
    
    # Calculate the numerator part for the second term
    numerator_2 = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator_2 *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * polygamma_term_2
    
    # First term
    term_1 = numerator_1 / (4 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    # Second term
    term_2 = numerator_2 / (4 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    # Final result
    result = term_1 - term_2
    
    return result

def deltathree(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Calculate the Polygamma functions
    polygamma_term_1 = polygamma(0, (gamma_value + delta_value - 2) / 2)
    polygamma_term_2 = polygamma(0, (gamma_value + delta_value) / 2)
    
    # Calculate the denominator components
    gamma_half_5 = gamma(gamma_value / 2)
    gamma_half_6 = gamma(delta_value / 2)
    
    denom_1 = gamma((gamma_value + delta_value - 3) / 2) * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma((gamma_value + delta_value - 1) / 2)
    
    denom = denom_1 - denom_2
    
    # Calculate the numerator part for the first term
    numerator_1 = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator_1 *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * polygamma_term_1
    
    # Calculate the numerator part for the second term
    numerator_2 = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator_2 *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * polygamma_term_2
    
    # First term
    term_1 = numerator_1 / (4 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    # Second term
    term_2 = numerator_2 / (4 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_5 * gamma_half_6 * denom)
    
    # Final result
    result = term_1 + term_2
    
    return result

def deltafour(gamma_value, delta_value, beta_value):
    # Calculate the Gamma functions
    gamma_half_1 = gamma((gamma_value - 1) / 2)
    gamma_half_2 = gamma((delta_value - 1) / 2)
    gamma_half_3 = gamma((gamma_value + delta_value - 2) / 2)
    gamma_half_4 = gamma((gamma_value + delta_value) / 2)
    
    # Additional Gamma terms for the complex numerator
    gamma_half_5 = gamma((gamma_value + delta_value - 3) / 2)
    gamma_half_6 = gamma((gamma_value + delta_value - 1) / 2)
    
    # Polygamma terms
    polygamma_term_1 = polygamma(0, (gamma_value + delta_value - 3) / 2)
    polygamma_term_2 = polygamma(0, (gamma_value + delta_value - 2) / 2)
    polygamma_term_3 = polygamma(0, (gamma_value + delta_value - 1) / 2)
    polygamma_term_4 = polygamma(0, (gamma_value + delta_value) / 2)
    
    # Denominator Gamma terms
    gamma_half_7 = gamma(gamma_value / 2)
    gamma_half_8 = gamma(delta_value / 2)
    
    denom_1 = gamma_half_5 * gamma_half_4
    denom_2 = beta_value * gamma_half_3 * gamma_half_6
    
    denom = denom_1 - denom_2
    
    # Numerator calculation
    part_1 = 0.5 * gamma_half_5 * gamma_half_4 * polygamma_term_1
    part_2 = -0.5 * beta_value * gamma_half_3 * gamma_half_6 * polygamma_term_2
    part_3 = -0.5 * beta_value * gamma_half_3 * gamma_half_6 * polygamma_term_3
    part_4 = 0.5 * gamma_half_5 * gamma_half_4 * polygamma_term_4
    
    # Full numerator
    numerator = (gamma_value + delta_value - 5) * (gamma_value + delta_value - 2 - 2 * beta_value)
    numerator *= gamma_half_1 * gamma_half_2 * gamma_half_3 * gamma_half_4 * (part_1 + part_2 + part_3 + part_4)
    
    # Final result
    result = - numerator / (2 * np.sqrt(np.pi) * (delta_value - 3) * gamma_half_7 * gamma_half_8 * denom**2)
    
    return result

# print(deltafour(2.1,2.5,0.1))


def df_delta(gamma_value, delta_value, beta_value):
    first_part = deltaone(gamma_value, delta_value, beta_value)
    
    second_part = deltatwo(gamma_value, delta_value, beta_value)
    
    third_part = deltathree(gamma_value, delta_value, beta_value)
    
    fourth_part = deltafour(gamma_value, delta_value, beta_value)
    
    # Combine all the parts
    return first_part + second_part + third_part + fourth_part