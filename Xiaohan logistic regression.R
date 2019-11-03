rm(list=ls())
install.packages("ROCR")
install.packages("stats")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("caTools")
install.packages("leaps")
install.packages("caret")

library("stats")
library("ROCR")
library("ggplot2")
library("dplyr")
library("caTools")
library("leaps")
library("caret")

data<-read.csv("C:\\Users\\Lenovo X240\\Documents\\NUS 2019-2020 AY1\\DSA4211\\ufc-project\\preprocessed_data.csv",header=T)
y_if_Red<-as.numeric(c(data$Winner=='Red'))
x<-data[,3:160]
#remove constant and zero columns
predictor<-x[,apply(x, 2, function(col) { length(unique(col)) > 1 })]
#scale the predictors
predictor<-scale(predictor, center = T, scale = T)
# do PCA and select the most important predictors
ufc.pca <- prcomp(predictor, center = F,scale. = F)
pc<-summary(ufc.pca)
importance<-pc$importance
#find the smallest model cover 85% for the overall variance
abs<-as.array(abs(importance[3,]-0.85)<0.005)
pc_no<-which(abs=="TRUE")[1]
pc_no
x_new<-pc$x[,1:pc_no]
train_no=floor(length(y_if_Red)/10)*7
test_no=floor(length(y_if_Red)/10)*3
x_train<-x_new[1:train_no,]
x_test<-x_new[(train_no+1):(train_no+test_no),]
y_train<-y_if_Red[1:train_no]
y_test<-y_if_Red[(train_no+1):(train_no+test_no)]
# Model fitting
model <- glm(y_train ~.,family=binomial(link='logit'),data=as.data.frame(x_train))
model_sum_pca<-summary(model)


#Assessing the predictive ability of the model
fitted.results <- predict(model,newdata=as.data.frame(x_test))
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(y_test != fitted.results)
print(paste('Accuracy',1-misClasificError))

#ROC value
p <- predict(model, newdata=as.data.frame(x_test))
pr <- prediction(p, y_test)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)


auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc






######################select features using fwd#############
# forward selection with max pred = 50
fwd_train = data.frame(y_train, predictor[1:train_no,]) 
regfit_fwd <- regsubsets(y_train~., data=fwd_train, method="forward", nvmax=150)
summary(regfit_fwd)$outmat
reg_summary <- summary(regfit_fwd)

#plot to show how many pred we should use which is 36 from cp plot
par(mfrow=c(2,2), mar=c(4,4,1,1))
plot(reg_summary$rss, xlab="No. of Variables", ylab="RSS", type="o")
plot(reg_summary$adjr2 ,xlab="No. of Variables", ylab="Adjusted R2", type="o") 
(K <- which.max(reg_summary$adjr2))  # adjusted R2
points(K, reg_summary$adjr2[K], col="red", cex=2, pch=20)
plot(reg_summary$cp ,xlab="No. of Variables", ylab="Cp", type="o")
(K <- which.min(reg_summary$cp))   # Cp
points(K, reg_summary$cp[K], col="red", cex=2, pch=20)
plot(reg_summary$bic ,xlab="No. of Variables", ylab="BIC", type="o")
(K <- which.min(reg_summary$bic))  # BIC
points(K, reg_summary$bic[K], col="red", cex=2, pch=20)

par(mfrow=c(1,3))
plot(regfit_fwd, scale="adjr2")
plot(regfit_fwd, scale="Cp")
plot(regfit_fwd, scale="bic")
coef<-as.data.frame(coef(regfit_fwd,36))
forward_pred<-(rownames(coef))[2:length(coef$`coef(regfit_fwd, 36)`)]
x<-data[,3:160]
#remove constant and zero columns
predictor<-x[,apply(x, 2, function(col) { length(unique(col)) > 1 })]
#scale the predictors
predictor<-scale(predictor, center = T, scale = T)
predictor_new<-predictor[,forward_pred]

x_train<-predictor_new[1:train_no,]
x_test<-predictor_new[(train_no+1):(train_no+test_no),]
# Model fitting
model <- glm(y_train ~.,family=binomial(link='logit'),data=as.data.frame(x_train))
model_sum_fwd<-summary(model)


#Assessing the predictive ability of the model
fitted.results <- predict(model,newdata=as.data.frame(x_test))
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(y_test != fitted.results)
print(paste('Accuracy',1-misClasificError))

#ROC value
p <- predict(model, newdata=as.data.frame(x_test))
pr <- prediction(p, y_test)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
