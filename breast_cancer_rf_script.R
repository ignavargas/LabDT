library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)

setwd("~/LabDT")
dir()

names <- c('id_number', 'diagnosis', 'radius_mean', 
           'texture_mean', 'perimeter_mean', 'area_mean', 
           'smoothness_mean', 'compactness_mean', 
           'concavity_mean','concave_points_mean', 
           'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 
           'area_se', 'smoothness_se', 'compactness_se', 
           'concavity_se', 'concave_points_se', 
           'symmetry_se', 'fractal_dimension_se', 
           'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 
           'smoothness_worst', 'compactness_worst', 
           'concavity_worst', 'concave_points_worst', 
           'symmetry_worst', 'fractal_dimension_worst')

breast_cancer <- read.table("breast_cancer.data", 
                            sep = ',', 
                            col.names = names)

row.names(breast_cancer) <- breast_cancer[,1]
breast_cancer[,1] <-  NULL

head(breast_cancer)

breast_cancer %>% 
  str()

breast_cancer %>% 
  dim()

#------ Chunk to check class imbalance
#------ Here we will discuss with the class the impact of Class imbalance

breast_cancer %>% 
  count(diagnosis) %>%
  group_by(diagnosis) %>%
  summarize(perc_dx = round((n / 569)* 100, 2))

#-------- It is not imbaanced (2:1) is a certain imbakance but it is not 10:1

#-------- Summary stats
summary(breast_cancer)

set.seed(2600)
trainIndex <- createDataPartition(breast_cancer$diagnosis, 
                                  p = .8, 
                                  list = FALSE, 
                                  times = 1)

training_set <- breast_cancer[ trainIndex, ]
test_set <- breast_cancer[ -trainIndex, ]

#---- The following is a custom grid search
#---- It is designed to search for custom parameters
#---- These parameters and the search will be done in the search
#---- for the DT, in this case the RF

customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, nodesize=param$nodesize, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

#---- We now have what we need to do the model training

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 3, 
  ## repeated ten times
  repeats = 10)

grid <- expand.grid(.mtry=c(floor(sqrt(ncol(training_set))), (ncol(training_set) - 1), floor(log(ncol(training_set)))), 
                    .ntree = c(100, 300, 500, 1000),
                    .nodesize =c(1:4))
set.seed(42)
fit_rf <- train(as.factor(diagnosis) ~ ., 
                data = training_set, 
                method = customRF, 
                metric = "Accuracy", 
                tuneGrid= grid,
                trControl = fitControl)

fit_rf$finalModel

#----- Variable Importance

varImportance <- varImp(fit_rf, scale = FALSE)

varImportanceScores <- data.frame(varImportance$importance)

varImportanceScores <- data.frame(names = row.names(varImportanceScores), var_imp_scores = varImportanceScores$B)

varImportanceScores

#----- Now visualize variable importance with ggplot

ggplot(varImportanceScores,
aes(reorder(names, var_imp_scores), var_imp_scores)) + 
  geom_bar(stat='identity', 
           fill = '#875FDB') + 
  theme(panel.background = element_rect(fill = '#fafafa')) + 
  coord_flip() + 
  labs(x = 'Feature', y = 'Importance') + 
  ggtitle('Feature Importance for Random Forest Model')


# ---- OOB Error, which reprents the 1/3 not used for training

oob_error <- data.frame(mtry = seq(1:100), oob = fit_rf$finalModel$err.rate[, 'OOB'])

paste0('Out of Bag Error Rate for model is: ', round(oob_error[100, 2], 4))

# ----- PLot OOB Error

ggplot(oob_error, aes(mtry, oob)) +  
  geom_line(colour = 'red') + 
  theme_minimal() + 
  ggtitle('OOB Error Rate across 100 trees') + 
  labs(y = 'OOB Error Rate')

predict_values <- predict(fit_rf, 
                          newdata = test_set)

ftable(predict_values, 
       test_set$diagnosis)

confusionMatrix(predict_values,
                test_set$diagnosis)
