library(caret)

resetTrain<-function(){
#CKD<-read.csv("LogisticRegressionData_5000.csv",stringsAsFactors = TRUE)
CKD<-read.csv("LogisticRegressionData_5000.csv")
nona.CKD<-CKD[complete.cases(CKD),]#remove na
#nona.CKD <- dummy.data.frame(nona.CKD, sep = ".")
#nona.CKD$CKD<-as.factor(nona.CKD$CKD)
nona.CKD0<-nona.CKD[nona.CKD$CKD==0,]
nona.CKD1<-nona.CKD[nona.CKD$CKD==1,]
trainsize<- floor(154)
train_ind0 <- sample(nrow(nona.CKD0), size = trainsize)#get indexes
train_ind1 <- sample(nrow(nona.CKD1), size = trainsize)#get indexes
train0 <- nona.CKD0[train_ind0, ][1:154,]
train1 <- nona.CKD1[train_ind1, ]
train<-rbind(train0,train1)
test0 <- nona.CKD0[-train_ind0,][1:67,]
test1 <- nona.CKD1[-train_ind1,]
test<-rbind(test0,test1)
}
resetTrain()

correlationMatrix <- cor(train)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
highlyCorrelated
names(train)[highlyCorrelated]


logtrain<-glm(CKD ~.,family="binomial",data=train)#0.9393064
#logtrain<-step(logtrain)
summary(logtrain)

#Signigicant on the ff:
logtrain<-glm(CKD ~
                +Age
                +I(Waist/Height)
                +Hypertension
                +Diabetes
                +CVD
              ,family="binomial",data=train)
              #,family="binomial",data=CKD2)
summary(logtrain)

modelaccuracy()
confusionmat()

resetTrain()
train[1,1]
print(train_ind[1])

modelaccuracy<-function(){
  levels<-c(.6,.575,.55,.525,.5,.475,.45,.425,.4,.3,.2,.1,0)
  #levels<-.5
  for(i in 1:length(levels)){    
    prdtest<-predict(logtrain,test,type="response")
    prdtest[prdtest >=levels[i]] <- 1
    prdtest[ prdtest != 1] <- 0
    prdtest[is.na(prdtest)] <- 0
    print(paste("Level:",levels[i]))
    print(table(prdtest, test$CKD))
    x<<-table(prdtest, test$CKD)
    correct<-(x[1,1]+x[2,2])/(sum(x))
    #print(paste(levels[i],"Level of:",correct))
  }
}

modelaccuracy<-function(){
  levels<-c(.6,.575,.55,.525,.5,.475,.45,.425,.4,.375,.35,.325,.3,.2,.1)
  #levels<-.5
  for(i in 1:length(levels)){    
    prdtest<-predict(logtrain,test,type="response")
    prdtest[prdtest >=levels[i]] <- 1
    prdtest[ prdtest != 1] <- 0
    prdtest[is.na(prdtest)] <- 0
    print(paste("Level:",levels[i]))
    print(table(prdtest, test$CKD))
    x<<-table(prdtest, test$CKD)
    correct<-(x[1,1]+x[2,2])/(sum(x))
    #print(paste(levels[i],"Level of:",correct))
  }
}
modelaccuracy()


prdtest<-predict(logtrain,test,type="response")
confusionMatrix(data = prdtest, test$CKD)

confusionMatrix(x)



prdtest<-predict(logtrain,nona.CKD,type="response")
prdtest[prdtest >=.5] <- 1
prdtest[ prdtest != 1] <- 0
prdtest[is.na(prdtest)] <- 0
print(table(prdtest, nona.CKD$CKD))

CKD2<-CKD[,c(2,3,14,11,25,27,30,24)]
CKD2<-CKD[complete.cases(CKD2),]#remove na

prdtest<-predict(logtrain,CKD2,type="response")
prdtest[prdtest >=.5] <- 1
prdtest[ prdtest != 1] <- 0
prdtest[is.na(prdtest)] <- 0
print(table(prdtest, CKD2$CKD))

modelaccuracy<-function(){
  levels<-c(.6,.575,.55,.525,.5,.475,.45,.425,.4,.375,.35,.325,.3,.2,.1)
  #levels<-.5
  for(i in 1:length(levels)){    
    prdtest<-predict(logtrain,CKD2,type="response")
    prdtest[prdtest >=levels[i]] <- 1
    prdtest[ prdtest != 1] <- 0
    prdtest[is.na(prdtest)] <- 0
    print(paste("Level:",levels[i]))
    print(table(prdtest, CKD2$CKD))
    x<<-table(prdtest, CKD2$CKD)
    correct<-(x[1,1]+x[2,2])/(sum(x))
    #print(paste(levels[i],"Level of:",correct))
  }
}
modelaccuracy()

############################Unbalanced
CKD<-read.csv("LogisticRegressionData_5000.csv")
CKD<-CKD[,c(2,3,14,11,24,25,27,30,34)]
nona.CKD <- CKD[complete.cases(CKD),]#remove na
nona.CKD <- dummy.data.frame(nona.CKD, sep = ".")
#nona.CKD$CKD <- as.factor(nona.CKD$CKD)
trainsize<- floor(2832)
train_ind <- sample(nrow(nona.CKD0), size = trainsize)#get indexes
train1 <- nona.CKD[train_ind, ]
test1<- nona.CKD[-train_ind,]

logtrain<-glm(CKD ~
                +Age
              +I(Waist/Height)
              +Hypertension
              +Diabetes
              +CVD
              
              ,family="binomial",data=train)
summary(logtrain)

modelaccuracy<-function(){
  levels<-c(.6,.575,.55,.525,.5,.475,.45,.425,.4,.375,.35,.325,.3,.2,.1)
  for(i in 1:length(levels)){    
    prdtest<-predict(logtrain,nona.CKD,type="response")
    prdtest[prdtest >=levels[i]] <- 1
    prdtest[ prdtest != 1] <- 0
    prdtest[is.na(prdtest)] <- 0
    print(paste("Level:",levels[i]))
    print(table(prdtest, nona.CKD$CKD))
  }
}
modelaccuracy()


CKD1000<-read.csv("LogisticRegressionData_1000.csv")
prdtest<-predict(logtrain,CKD1000,type="response")
prdtest[prdtest >=.325] <- 1
prdtest[ prdtest != 1] <- 0
prdtest[is.na(prdtest)] <- 0
sum(prdtest)




