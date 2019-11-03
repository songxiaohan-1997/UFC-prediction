#################################################
## High-Dimensional Statistical Analysis Project
## 
## Joel & Xiohan & Jeppe & Ragnhild
## November 2019
#################################################

# Libraries used 
library(tidyverse)
library(caret)
library(glmnet)
library(MASS)

set.seed(1234) 
data = read.csv("preprocessed_data.csv",header=TRUE) # load the data

sum(sapply(data, function(x) sum(is.na(x)))) #=0     # checking there is no NA values

data_y = 1*(data$Winner == "Red") # response, change to 0 for Blue and 1 for red
data$title_bout = as.numeric(data$title_bout) - 1 # Change to dummy variable 
data_x = data %>% dplyr::select(-Winner) # features

## Summary of response and 
summary(cbind(data_y,data$R_age, data$R_losses, data$R_avg_opp_SIG_STR_landed, data$B_Height_cms))

n = dim(data)[1]        # no of obs in total
n_train = floor(0.70*n) # no of obs in training set
n_test = n-n_train      # no of obs in test set 

train_idx = sample(1:n,n_train)   # randomly assigning n_train obs to traning set
test_idx = setdiff(1:n,train_idx) # assigning the rest to test set

## -------------------------------------------------------------------------------
## Processing the data
## -------------------------------------------------------------------------------

train_x = data_x[train_idx,] # training observations features used to process data

## Keeping non-constant features in training set 
non_constant_idx = as.vector(apply(data.matrix(train_x), 2, var) != 0)
train_x.1 = train_x[,non_constant_idx] # 2 features was removed

## Removing highly correlated features from the remaining data 
corr_idx = findCorrelation(cor(data.matrix(train_x.1)), cutoff = 0.95)

# New data matrix of features is 
data_x = data_x[,non_constant_idx] # removing constant columns
data_x = data_x[,-corr_idx]        # removing highly correlated columns 

# Scaling
train_x = data_x[train_idx,]
scale_mean = colMeans(train_x)
scale_sd = apply(train_x, 2, sd)

data_x = (t(data.matrix(data_x)) - scale_mean ) / scale_sd
data_x = t(data_x)

## Data to use for analysis
train_y = data_y[train_idx]
train_x = data_x[train_idx,]

test_y = data_y[test_idx]
test_x = data_x[test_idx,]

data_new = cbind(data_y,data_x)
data_new = as.data.frame(data_new)

## --------------------------------------------------------------------------
## Logistic regresion 
## --------------------------------------------------------------------------

## Logistic regression on the whole dataset
fit_lr = glm(data_y~., data = data_new, family = binomial, subset = train_idx)

## Accuracy on training data
pred_train = 1*(predict(fit_lr, type = "response")>0.5)
table(pred_train,train_y)
mean(pred_train == train_y)

## Accuracy on test data
probs_test = predict(fit_lr, newdata = data_new[test_idx,], type = 'response')
pred_test = 1*(probs_test > 0.5)
table(pred_test,test_y)
mean(pred_test == test_y)


## Logistic regression with regularization 

# Cross validation to find optimal lambda
cv.fit_lr = cv.glmnet(data.matrix(train_x), train_y, family = "binomial", 
                   alpha = 1, nfolds = 10, keep = T)

plot(cv.fit_lr)

best_lambda = cv.fit_lr$lambda.1se

reg_lr = glmnet(data.matrix(train_x), train_y, family = "binomial", alpha = 1, lambda = best_lambda)


best = summary(reg_lr$beta)[,1] # the positive beta values
ext = best+1                    # columns to substract from the         

## Accuracy on training data
pred_train_reg = predict(reg_lr, newx = data.matrix(train_x), type = "class")
table(pred_train_reg,train_y)
mean(pred_train_reg == train_y)

## Accuracy on test 
pred_test_reg = predict(reg_lr, newx = data.matrix(test_x), type = "class")
table(pred_test_reg, test_y)
mean(pred_test_reg == test_y)

# Forward selection using lasso regularization
lambda = cv.fit_lr$lambda[2]
c = coef(cv.fit_lr,s=lambda,exact=TRUE)

plot(cv.fit_lr$glmnet.fit,xlim = c(0,0.3), ylim = c(-0.1,0.01), col = 1:200)

## --------------------------------------------------------------------------------
## Neural networks
train_y_temp = data.matrix(train_y)
test_y_temp = data.matrix(test_y)

dim(train_y_temp)
dim(test_y_temp)
dim(train_x)
dim(test_x)

library_list = c("ggplot2", "dplyr", "keras", "caTools", "leaps", "caret")
lapply(library_list, require, character.only = TRUE)

train_y = to_categorical(ifelse(train_y == 1, 1, 0))
test_y = to_categorical(ifelse(test_y == 1, 1, 0))

dim(train_y)
dim(test_y)
dim(train_x)
dim(test_x)

# run a ff neural network
input = layer_input( shape = ncol(train_x))
output = input %>%
  layer_dense(units = 50, activation = "elu") %>%
  layer_dense(units = 30, activation = "sigmoid") %>%
  layer_dense(units = 30, activation = "tanh") %>%
  layer_dense(units = 2, activation = "softmax") 

model = keras_model(inputs = input,
                    outputs = output)
model %>% summary  
model %>% compile(optimizer = "adam",
                  loss = "categorical_crossentropy",
                  metrics = c("accuracy")
)

learn.ff <- model %>% fit(x=train_x, y=train_y,
                          epochs = 40, 
                          batch_size = 128,
                          callbacks = list(
                            callback_reduce_lr_on_plateau(monitor = "val_loss",
                                                          min_delta = 1e-04, 
                                                          factor = 0.1,
                                                          patience = 5,
                                                          cooldown = 0, 
                                                          min_lr = 0)
                          ),
                          validation_split=0.2)

# Plot of model
plot(learn.ff)

# Computing predictions
model %>% evaluate(test_x, test_y)

# Confusion matrix
probs = model %>% predict(test_x)
probs[,2]

Direction <- factor(test_y_temp, labels=c("Blue", "Red"))
pred <- rep("Blue",length(test_y_temp))
pred[probs[,2] > 0.5] <- "Red"
table(pred, Direction) # pred: predictions, Direction: actual observations

mean(pred==Direction)  # proportion of correct predictions for training data
mean(pred!=Direction)  # training error rate
