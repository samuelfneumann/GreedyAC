
Remove auto-entropy for SAC Discrete

Removed the automatic entropy tuning implementation from SAC with
discrete actions. We do not use auto-entropy in our experiments, and it
actually introduces two new hyperparameters to tune in this scenario:

1. Entropy scale stepsize
2. Entropy target

whereas with a set entropy scale, we just have a single hyperparameter
to tune.

For continuous action environments, we can use the SAC author's
suggestion for determining the entropy scale target.
# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# On branch master
# Your branch is up to date with 'origin/master'.
#
# Changes to be committed:
#	modified:   agent/nonlinear/SAC.py
#	modified:   agent/nonlinear/SACDiscrete.py
#	new file:   config/agent/SACDiscrete.json
#	modified:   utils/experiment_utils.py
#
