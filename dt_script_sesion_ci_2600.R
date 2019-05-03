install.packages("rpart")
library(rpart)
library(caret)

setwd("~/Desktop/ml_bioinfo_i_2019/ejercicio_3/")
dir()

letter.rec.data <- read.table("letter-recognition.data",
                              header = FALSE,
                              sep = ",")

head(letter.rec.data)

#------------- Classification of letters using CART

set.seed(101)
letter.model <- rpart(V1 ~ .,
              data= letter.rec.data,
              method='class')

letter.model.pred <- predict(letter.model,
                             type='class')

mean(letter.model.pred == letter.rec.data$V1)

cart.conf.mat <- confusionMatrix(letter.model.pred,
                                 letter.rec.data$V1)

cart.conf.mat

#----------- PPrediction of letters with RFs

library(randomForest)
set.seed(101)
rf.letter.rec <- randomForest(letter.rec.data,
                              y = letter.rec.data$V1,
                              ntree = 750,
                              importance = TRUE)
rf.letter.rec$importance

letter.rf.pred <- predict(rf.letter.rec,
                             type='class')

mean(letter.rf.pred == letter.rec.data$V1)

conf.mat <- confusionMatrix(letter.rf.pred,
                            letter.rec.data$V1)

conf.mat
