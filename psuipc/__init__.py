"""PS-UIP: a propensity-score-calibrated unit information prior for borrowing
historical control information into a current two-arm randomized trial.

Modules
-------
dgm          : data-generating mechanism (current RCT + control-only sources)
stage1       : propensity-score design stage (membership PS, IPTW, overlap, summaries)
fast_sampler : blocked Gibbs / Polya-Gamma sampler for the analysis model
methods      : PS-UIP and the benchmark borrowing priors
run          : operating-characteristic Monte Carlo over the scenario grid
sweep        : conflict-magnitude sweep experiment
sensitivity  : conflict-discount sensitivity analysis
application  : ACTG175 real-data illustration
"""

__version__ = "1.0.0"
