# *****************************************************************************
# Lab 10: Ensemble Methods ----
#
# Course Code: BBT4206
# Course Name: Business Intelligence II
# Semester Duration: 21st August 2023 to 28th November 2023
#
# Lecturer: Allan Omondi
# Contact: aomondi [at] strathmore.edu
#
# Note: The lecture contains both theory and practice. This file forms part of
#       the practice. It has required lab work submissions that are graded for
#       coursework marks.
#
# License: GNU GPL-3.0-or-later
# See LICENSE file for licensing information.
# *****************************************************************************

# **[OPTIONAL] Initialization: Install and use renv ----
# The R Environment ("renv") package helps you create reproducible environments
# for your R projects. This is helpful when working in teams because it makes
# your R projects more isolated, portable and reproducible.

# Further reading:
#   Summary: https://rstudio.github.io/renv/
#   More detailed article: https://rstudio.github.io/renv/articles/renv.html

# "renv" It can be installed as follows:
# if (!is.element("renv", installed.packages()[, 1])) {
# install.packages("renv", dependencies = TRUE,
# repos = "https://cloud.r-project.org") # nolint
# }
# require("renv") # nolint

# Once installed, you can then use renv::init() to initialize renv in a new
# project.

# The prompt received after executing renv::init() is as shown below:
# This project already has a lockfile. What would you like to do?

# 1: Restore the project from the lockfile.
# 2: Discard the lockfile and re-initialize the project.
# 3: Activate the project without snapshotting or installing any packages.
# 4: Abort project initialization.

# Select option 1 to restore the project from the lockfile
# renv::init() # nolint

# This will set up a project library, containing all the packages you are
# currently using. The packages (and all the metadata needed to reinstall
# them) are recorded into a lockfile, renv.lock, and a .Rprofile ensures that
# the library is used every time you open the project.

# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)

# This can also be configured using the RStudio GUI when you click the project
# file, e.g., "BBT4206-R.Rproj" in the case of this project. Then
# navigate to the "Environments" tab and select "Use renv with this project".

# As you continue to work on your project, you can install and upgrade
# packages, using either:
# install.packages() and update.packages or
# renv::install() and renv::update()

# You can also clean up a project by removing unused packages using the
# following command: renv::clean()

# After you have confirmed that your code works as expected, use
# renv::snapshot(), AT THE END, to record the packages and their
# sources in the lockfile.

# Later, if you need to share your code with someone else or run your code on
# a new machine, your collaborator (or you) can call renv::restore() to
# reinstall the specific package versions recorded in the lockfile.

# [OPTIONAL]
# Execute the following code to reinstall the specific package versions
# recorded in the lockfile (restart R after executing the command):
# renv::restore() # nolint

# [OPTIONAL]
# If you get several errors setting up renv and you prefer not to use it, then
# you can deactivate it using the following command (restart R after executing
# the command):
# renv::deactivate() # nolint

# If renv::restore() did not install the "languageserver" package (required to
# use R for VS Code), then it can be installed manually as follows (restart R
# after executing the command):

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# In addition to hyperparameter tuning, you can also combine the predictions of
# multiple different models together. This is called an "ensemble prediction".

## Ensemble Methods ----
### (1) Bagging (Bootstrap Aggregation) ----
# Building multiple models (typically models of the same type) from different
# subsamples of the training dataset.

### (2) Boosting ----
# Building multiple models (typically models of the same type) each of which
# learns to fix the prediction errors of a prior model in the chain.

### (3) Stacking ----
# Building multiple models (typically models of differing types) and a
# supervised model that learns how to best combine the predictions of the
# primary models.

# STEP 1. Install and Load the Required Packages ----
## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caretEnsemble ----
if (require("caretEnsemble")) {
  require("caretEnsemble")
} else {
  install.packages("caretEnsemble", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## C50 ----
if (require("C50")) {
  require("C50")
} else {
  install.packages("C50", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## adabag ----
if (require("adabag")) {
  require("adabag")
} else {
  install.packages("adabag", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# STEP 2. Load the Dataset ----
data(Ionosphere)
dataset <- Ionosphere

# The second attribute is a constant and has been removed.
dataset <- dataset[, -2]

# The first attribute is a factor (0, 1) and has been transformed to be numeric
# for consistency with all of the other numeric attributes.
dataset$V1 <- as.numeric(as.character(dataset$V1))

# 1. Bagging ----
# Two popular bagging algorithms are:
## 1. Bagged CART
## 2. Random Forest

# Example of Bagging algorithms
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
seed <- 7
metric <- "Accuracy"

## 2.a. Bagged CART ----
set.seed(seed)
ion_model_bagged_cart <- train(Class ~ ., data = dataset, method = "treebag",
                               metric = metric,
                               trControl = train_control)

## 2.b. Random Forest ----
set.seed(seed)
ion_model_rf <- train(Class ~ ., data = dataset, method = "rf",
                      metric = metric, trControl = train_control)

# Summarize results
bagging_results <-
  resamples(list("Bagged Decision Tree" = ion_model_bagged_cart,
                 "Random Forest" = ion_model_rf))

summary(bagging_results)
dotplot(bagging_results)

# 2. Boosting ----
# Three popular boosting algorithms are:
## 1. AdaBoost.M1
## 2. C5.0
## 3. Stochastic Gradient Boosting

# Example of Boosting Algorithms
train_control <- trainControl(method = "cv", number = 5)
seed <- 7
metric <- "Accuracy"

## 1.a. Boosting with C5.0 ----
# C5.0
set.seed(seed)
ion_model_c50 <- train(Class ~ ., data = dataset, method = "C5.0",
                       metric = metric,
                       trControl = train_control)

## 1.b. Boosting with Stochastic Gradient Boosting ----
set.seed(seed)
ion_model_gbm <- train(Class ~ ., data = dataset, method = "gbm",
                       metric = metric, trControl = train_control,
                       verbose = FALSE)

## 1.c. Boosting with AdaBoost.M1 ----
set.seed(seed)

# NB: This will take a long time to execute
ion_model_adaboost <- train(Class ~ ., data = dataset, method = "AdaBoost.M1",
                            metric = metric, trControl = train_control)

# Summarize results
boosting_results <-
  resamples(list("C5.0 boosting ensemble" = ion_model_c50,
                 "Generalized Boosted Regression Modeling (GBM)" =
                   ion_model_gbm,
                 "AdaBoost" = ion_model_adaboost))

summary(boosting_results)
dotplot(boosting_results)

# 3. Stacking ----
# The "caretEnsemble" package allows you to combine the predictions of multiple
# caret models.

## caretEnsemble::caretStack() ----
# Given a list of caret models, the "caretStack()" function (in the
# "caretEnsemble" package) can be used to specify a higher-order model to
# learn how to best combine together the predictions of sub-models.

## caretEnsemble::caretList() ----
# The "caretList()" function provided by the "caretEnsemble" package can be
# used to create a list of standard caret models.

# Example of Stacking algorithms
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                              savePredictions = TRUE, classProbs = TRUE)
set.seed(seed)

algorithm_list <- c("glm", "lda", "rpart", "knn", "svmRadial")
models <- caretList(Class ~ ., data = dataset, trControl = train_control,
                    methodList = algorithm_list)

# Summarize results before stacking
results <- resamples(models)
summary(results)
dotplot(results)

# The predictions made by the sub-models that are combined using stacking
# should have a low-correlation (for diversity amongst the sub-models, i.e.,
# different sub-models are accurate in different ways). If the predictions for
# the sub-models were highly correlated (> 0.75) then they would be making the
# same or very similar predictions most of the time reducing the benefit of
# combining the predictions.

# correlation between results
modelCor(results)
splom(results)

## 3.a. Stack using glm ----
stack_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                              savePredictions = TRUE, classProbs = TRUE)
set.seed(seed)
ion_model_stack_glm <- caretStack(models, method = "glm", metric = "Accuracy",
                                  trControl = stack_control)
print(ion_model_stack_glm)

## 3.b. Stack using random forest ----
set.seed(seed)
ion_model_stack_rf <- caretStack(models, method = "rf", metric = "Accuracy",
                                 trControl = stack_control)
print(ion_model_stack_rf)

# [OPTIONAL] **Deinitialization: Create a snapshot of the R environment ----
# Lastly, as a follow-up to the initialization step, record the packages
# installed and their sources in the lockfile so that other team-members can
# use renv::restore() to re-install the same package version in their local
# machine during their initialization step.
# renv::snapshot() # nolint