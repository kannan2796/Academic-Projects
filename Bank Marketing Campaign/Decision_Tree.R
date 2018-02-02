library(readr)
bankfull <- read_csv("C:/Users/karth/Desktop/DESKTOP_1/CLASS NOTES/Q3/Data Mining/Project/Bank/bank-full.csv")
View(bankfull)

library(tree)
library(caret)

bankf <- bankfull

bankf$age <- as.numeric(bankfull$age)
bankf$job <- as.factor(bankfull$job)
bankf$marital <- as.factor(bankfull$marital)
bankf$education <- as.factor(bankfull$education)
bankf$default <- as.factor(bankfull$default)
bankf$balance <- as.numeric(bankfull$balance)
bankf$housing <- as.factor(bankfull$housing)
bankf$loan <- as.factor(bankfull$loan)
bankf$day <- as.numeric(bankfull$day)
bankf$month <- as.factor(bankfull$month)
bankf$duration <- as.numeric(bankfull$duration)
bankf$campaign <- as.numeric(bankfull$campaign)
bankf$pdays <- as.numeric(bankfull$pdays)
bankf$previous <- as.numeric(bankfull$previous)
bankf$poutcome <- as.factor(bankfull$poutcome)
bankf$y <- as.factor(bankfull$y)



#To Split Test and train


smp_size <- floor(0.75 * nrow(bankf))


set.seed(123)

train_bank <- sample(seq_len(nrow(bankf)), size = smp_size)

train <- bankf[train_bank, ]
test <- bankf[-train_bank, ]




#Modeling with Training set


banktree <- tree(y~., data = train, method = "class")

plot(banktree)

text(banktree)




#Performing Cross Vlidation to determine the size of the tree

set.seed(2)

cvbank <- cv.tree(banktree, K=10)

cvbank

plot(cvbank)
plot(cvbank, pch=21, bg=8, type="p", cex=1.5, ylim=c(22000,35000))

#Based on CV we could see that a size of 8 trees has the minimal deviance, 
#lets build back the tree with 8 trees


#We will start with pruning the tree
bankcut <- prune.tree(banktree)

bankcut

plot(bankcut)

bankprune <- prune.tree(banktree, best = 8)

plot(bankprune)

text(bankprune)


#To Predict the model


model1 = predict(bankprune, test, type = "class")

model1

length(model1)


confusionMatrix(model1,test$y)

recall(model1,test$y)

precision(model1,test$y)

F_meas(model1,test$y)







