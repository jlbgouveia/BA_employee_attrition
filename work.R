library(data.table) 
library(ggplot2)    
library(caret)
library(entropy)    
library(mlbench)    
library(partykit)   
library(foreign)    
library(rpart)
library(rpart.plot)
library(ROCR)
library(stargazer)
library(scales)
library(ggthemes)

rm(list = ls())
setwd("C:/Users/Eduardo/Google Drive/Católica/BA/BA Group work/Working directory")

data    <- read.csv("train.csv", header = TRUE)
dt.train<- data.table(data)
rm(data)

desc<- read.csv("variabledefinition.txt", header = F)

#-------------------


# Check fot missing data 
as.matrix(sapply(dt.train, function(x) sum(is.na(x))))

#removing the variable  percentsalaryhike
dt.train[, percentsalaryhike := NULL] 

#Checking for class inbalance
dt.train[, .N , by = attrition ]
(dt.train[, sum(attrition=="Yes")]/nrow(dt.train))*100 #14.7% of workers left 1 year after the information was gathered

#checking for variance in indicators 
nzvar     <- nearZeroVar(dt.train, saveMetrics = TRUE) #check for near zero variance
nzvremove <- rownames(nzvar)[ nzvar$nzv ]              #get the names of those variables
nzvremove

#---------------------

new.data<- read.csv("prediction.csv", header = TRUE)
dt.prediction<- data.table(new.data)
rm(new.data)

#split the train dataset

set.seed(123)
train.ix <- sample(1:nrow(dt.train), size = 0.8 * nrow(dt.train), replace = FALSE)
dt.train.train <- dt.train[train.ix,] 
dt.test.train  <- dt.train[-train.ix,] 

# Designing the models 

#Training the model

fivestats         <- function(...) c(twoClassSummary(...), defaultSummary(...))
opts.trainning.cv <-
  trainControl(
    method            = 'cv' #cross validation
    , number          = 5   #number of folds
    , classProbs      = TRUE
    , verboseIter     = TRUE    #output model training details
    , savePredictions =  TRUE #keep all the out of sample predictions
    , summaryFunction = fivestats)

#Single classification tree

#Creation
set.seed(123)
system.time(x.model.rpart <- train(
  attrition ~ .
  , data      =  dt.train.train
  , method    = "rpart"
  , trControl = opts.trainning.cv
  , tuneGrid  = expand.grid(cp = c(0.0000001, 0.000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, .25 ,.3, 0.5)) 
  , preProc   = c()
  , metric    = "ROC"))

#Evaluation
plot(x.model.rpart)
confusionMatrix(x.model.rpart) #85.4% accuracy
rpart.pred <- predict(x.model.rpart    , newdata = dt.test.train, type = 'prob')
pred.rpart <- prediction(rpart.pred[,1], dt.test.train$attrition, label.ordering = c('Yes','No'))
perf.rpart <- performance(pred.rpart   , "tpr", "fpr")

plot(perf.rpart,colorize=T)
points(x=seq(0,1,0.1),y=seq(0,1,0.1),type='line',col='navy')
(as.numeric(performance(pred.rpart,"auc")@y.values))*100 #AUC=72%


#Logistic Regression

#Creation 

set.seed(123)
system.time(
  x.model.glmnet <- train(
    attrition ~  .
    , data      =  dt.train.train
    , method    = "glmnet"
    , trControl = opts.trainning.cv
    , tuneGrid  = expand.grid(alpha = c(0, 0.1, 0.2, 0.5, 1), lambda = c(0.1,0.2,0.5,1)) 
    , preProc   = c('center','scale')
    , metric    = "ROC"))

#Evaluation 

plot(x.model.glmnet)
confusionMatrix(x.model.glmnet) #87% accuracy
glmnet.pred <- predict(x.model.glmnet    , newdata = dt.test.train, type = 'prob')
pred.glmnet <- prediction(glmnet.pred[,1], dt.test.train$attrition, label.ordering = c('Yes','No')) #error here
perf.glmnet <- performance(pred.glmnet   , "tpr", "fpr")

plot(perf.glmnet,colorize=T)
points(x=seq(0,1,0.1),y=seq(0,1,0.1),type='line',col='navy')
(as.numeric(performance(pred.glmnet,"auc")@y.values))*100 #AUC=79.8%


#Random Forest

#Creation
set.seed(123)
system.time(
  x.model.rf <- train(
    attrition ~  .
    , data      =  dt.train.train
    , method    = "rf"
    , trControl = opts.trainning.cv
    , tuneGrid  = expand.grid( mtry = c(2,7,15)) 
    , preProc   = c()
    , metric    = "ROC"))

#Evaluation

plot(x.model.rf)
confusionMatrix(x.model.rf) #88.5% accuracy
rf.pred <- predict(x.model.rf       , newdata = dt.test.train, type = 'prob')
pred.rf <- prediction(rf.pred[,1]   , dt.test.train$attrition, label.ordering = c('Yes','No'))
perf.rf <- performance(pred.rf   , "tpr", "fpr")


#K-nearest neighbours

set.seed(123)
system.time(
  x.model.knn <- train(
    attrition ~  .
    , data      =  dt.train.train
    , method    = "knn"
    , trControl = opts.trainning.cv
    , tuneGrid  = expand.grid(k = c(5, 11, 25,  101, 125 ,151, 175)) #choosing odd numbers 
    , preProc   = c('center','scale')
    , metric    = "ROC"))

plot(x.model.knn)
confusionMatrix(x.model.knn) #accuracy=86.8%
knn.pred <- predict     (x.model.knn    , newdata = dt.test.train, type = 'prob')
pred.knn <- prediction  (knn.pred[,1], dt.test.train$attrition, label.ordering = c('Yes','No'))
perf.knn <- performance (pred.knn   , "tpr", "fpr")

#Evaluation of all models

plot(perf.rpart , col ='Blue')
plot(perf.glmnet, col ='Orange' , add = TRUE)
plot(perf.knn   , col ='Purple' , add = TRUE)
plot(perf.rf    , col ='Red'    , add = TRUE)
points(x=seq(0,1,0.1),y=seq(0,1,0.1),type='l',col='navy')
legend(  'bottomright'
         , legend = c('rpart', 'glmnet','Knn' ,'rforest')
         , col    = c('Blue' , 'Orange','Purple','Red'    )
         , lty    = c(1,1,1,1))

#Measuring AUC
(as.numeric(performance(pred.rpart, "auc")@y.values))*100  #AUC=72.0%
(as.numeric(performance(pred.glmnet, "auc")@y.values))*100 #AUC=79.8%
(as.numeric(performance(pred.knn , "auc")@y.values))*100   #AUC=78.0%
(as.numeric(performance(pred.rf , "auc")@y.values))*100    #AUC=78.8%

#Converting models' scores into probabilities

#Checking the models' calibration
dt.calibration <- data.table(
  attrition    =  dt.test.train$attrition
  ,  mrpart    = rpart.pred[  ,'No']
  ,  mglmnet   = glmnet.pred[ ,'No']
  ,  mknn      = knn.pred[    ,'No']
  ,  mrf       = rf.pred[     ,'No'])

ggplot(calibration( attrition ~ mrpart , data = dt.calibration)) + ylim(0,100) + theme_bw()
ggplot(calibration( attrition ~ mglmnet, data = dt.calibration)) + ylim(0,100) + theme_bw()
ggplot(calibration( attrition ~ mknn   , data = dt.calibration)) + ylim(0,100) + theme_bw()
ggplot(calibration( attrition ~ mrf    , data = dt.calibration)) + ylim(0,100) + theme_bw()

#Calibrating the models' probabilities

dt.calibration <- data.table(
  attrition  = dt.test.train$attrition,
  mrpart     = predict(x.model.rpart,  newdata = dt.test.train, type = 'prob')[,'No'],
  mglmnet    = predict(x.model.glmnet, newdata = dt.test.train, type = 'prob')[,'No'],
  mknn       = predict(x.model.knn,    newdata = dt.test.train, type = 'prob')[,'No'],
  mrf        = predict(x.model.rf,     newdata = dt.test.train, type = 'prob')[,'No'])

dt.calibration[,flg_attrition := as.double(ifelse(attrition=='No',1,0))]

#Building calibration model
mrpart.calibrated  <- glm( flg_attrition ~ mrpart,  dt.calibration,  family = binomial(link='logit'))
mglmnet.calibrated <- glm( flg_attrition ~ mglmnet, dt.calibration,  family = binomial(link='logit'))
mknn.calibrated    <- glm( flg_attrition ~ mknn,    dt.calibration,  family = binomial(link='logit'))
mrf.calibrated     <- glm( flg_attrition ~ mrf,     dt.calibration,  family = binomial(link='logit'))


#Final models' probabilities
dt.calibration[, mrpart_cal  := predict( mrpart.calibrated, newdata  = dt.calibration,type = 'response')]
dt.calibration[, mglmnet_cal := predict( mglmnet.calibrated, newdata = dt.calibration,type = 'response')]
dt.calibration[, mknn_cal    := predict( mknn.calibrated, newdata    = dt.calibration,type = 'response')]
dt.calibration[, mrf_cal     := predict( mrf.calibrated, newdata     = dt.calibration,type = 'response')]


#Checking if the calibrated probabilites are real probabilities
ggplot(calibration(attrition ~ mrpart_cal,  data = dt.calibration)) + ylim(0,100) +  theme_bw()
ggplot(calibration(attrition ~ mglmnet_cal, data = dt.calibration)) + ylim(0,100) +  theme_bw()
ggplot(calibration(attrition ~ mknn_cal,    data = dt.calibration)) + ylim(0,100) +  theme_bw()
ggplot(calibration(attrition ~ mrf_cal,     data = dt.calibration)) + ylim(0,100) +  theme_bw()

#--------------------------------------------------------------------------------

#Profit Modelling

#Scores 

score.rpart <- predict(object = x.model.rpart,  newdata = dt.prediction, type = 'prob')
score.glmnet<- predict(object = x.model.glmnet, newdata = dt.prediction, type = 'prob')
score.knn   <- predict(object = x.model.knn,    newdata = dt.prediction, type = 'prob')
score.rf    <- predict(object = x.model.rf,     newdata = dt.prediction, type = 'prob')

dt.prediction[, mrpart  := score.rpart$No]
dt.prediction[, mglmnet := score.glmnet$No]
dt.prediction[, mknn    := score.knn$No]
dt.prediction[, mrf     := score.rf$No]

#Probabilities

dt.prediction [, mrpart.pred    := predict ( mrpart.calibrated,  newdata = dt.prediction,type = 'response')]
dt.prediction [, mglmnet.pred   := predict ( mglmnet.calibrated, newdata = dt.prediction,type = 'response')]
dt.prediction [, mknn.pred      := predict ( mknn.calibrated,    newdata = dt.prediction,type = 'response')]
dt.prediction [, mrf.pred       := predict ( mrf.calibrated,     newdata = dt.prediction,type = 'response')]

correlation.matrix <- round(cor(dt.prediction[,c("mrpart.pred","mglmnet.pred", "mknn.pred","mrf.pred")]),2) 
head(correlation.matrix) #checking if the models' predictions are aligned 

#Expected values

#Target
dt.prediction[,  profit:= utilization_billed - monthlyincome]
dt.prediction[, expected.target :=  profit - (0.2*monthlyincome)]

#Not target 
dt.prediction[, expected.not.target.mrpart  := mrpart.pred * profit]
dt.prediction[, expected.not.target.mglmnet := mglmnet.pred * profit]
dt.prediction[, expected.not.target.mknn    := mknn.pred * profit]
dt.prediction[, expected.not.target.mrf     := mrf.pred * profit]

#Computing differences
dt.prediction[, expected.value.mrpart  := expected.target - expected.not.target.mrpart]
dt.prediction[, expected.value.mglmnet := expected.target - expected.not.target.mglmnet]
dt.prediction[, expected.value.mknn    := expected.target - expected.not.target.mknn]
dt.prediction[, expected.value.mfr     := expected.target - expected.not.target.mrf]


dt.prediction <- dt.prediction[order(-expected.value.mglmnet)]
dt.prediction[, profit.mglmnet := cumsum(expected.value.mglmnet*12) ]
dt.prediction[, investment.mglmnet:= cumsum(monthlyincome*0.2*12)]

ggplot(data = dt.prediction, aes(x = investment.mglmnet , y = profit.mglmnet)) +
  geom_line() +
  geom_vline(xintercept = 1000000, color = 'red', linetype = 'longdash') +
  geom_hline(yintercept = dt.prediction[,max(profit.mglmnet)], color = 'navy', linetype = 'longdash') +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(labels = comma, breaks = seq(0,8000000, by=1000000) ) + 
  labs( x = "Investment", y = "Profit", title = "Investment plan using the Logistic Regression Model") +
  theme_few() +
  coord_cartesian( xlim = c(0, 5000000), ylim = c(0, 9000000)) 

for (employee in 1:nrow(dt.prediction)) {
  if(dt.prediction[employee,investment.mglmnet] > 1000000) {
    
  break()
    }
}


employee-1 # To maximize the profit within the budget restricition, the company would need to target the 117 most profitable employees.
dt.prediction[employee-1,investment.mglmnet] # The correspondent cumulative investiment totals to 984475.2.
dt.prediction[employee-1, profit.mglmnet] #The total profit brought by the investment equals 7,656,456.
dt.prediction[, max(profit.mglmnet)] #If there were no budget restrictions, the total profit seized would have been totaled 8,139,994.

#----------------------------------

#Competition data set 
rm(dt.competition)

dt.competition <- data.table(
  employee.number        = dt.prediction$employeenumber,
  attrition.probability  = dt.prediction$mglmnet.pred,
  monthly.income         = dt.prediction$monthlyincome,
  expected.profit        = dt.prediction$expected.value.mglmnet*12,
  incentive.amount       = (dt.prediction$monthlyincome)*0.5*12
  )

dt.competition[,investment         := cumsum(monthly.income*0.5*12)]
dt.competition[,retention.decision := ifelse(investment<1000000, "Invest", "Don't invest")]
dt.competition[,investment := NULL]
dt.competition[, monthly.income    :=NULL]

write.csv(dt.competition, "dt.competition.csv")


