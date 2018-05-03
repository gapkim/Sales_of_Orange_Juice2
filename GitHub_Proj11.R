##############################
### Initial Setup

knitr::opts_chunk$set(comment=NA, echo=FALSE, warning=TRUE, message=FALSE,
                      fig.align="center")
options(digits=4)
rm(list=ls())

library(ISLR)
data(OJ)

#################################
### EXPLORATORY DATA ANALYSIS
#################################

OJ$StoreID = as.factor(OJ$StoreID)
OJ$SpecialCH = as.factor(OJ$SpecialCH)
OJ$SpecialMM = as.factor(OJ$SpecialMM)
OJ$STORE = as.factor(OJ$STORE)

### Table 1: Total count of factors in categorical variables

library(htmlTable)

tab1 = c(table(OJ$Purchase), table(OJ$SpecialCH), table(OJ$SpecialMM), table(OJ$Store7))
tab2 = c(table(OJ$StoreID), table(OJ$STORE))

htmlTable(tab1,
          caption="Table 1: Total count of factors in categorical variables",
          header = names(tab1),
          cgroup = c("Purchase","SpecialCH","SpecialMM", "Store7"),
          n.cgroup = c(2,2,2,2),
          css.cell = "width: 60px;")

htmlTable(tab2,
          caption="Table 1(continued): Total count of factors in categorical variables",
          header = names(tab2),
          cgroup = c("StoreID","STORE"),
          n.cgroup = c(5,5),
          css.cell = "width: 60px;")

### Figure 1: Boxplots of quantitative variables

library(reshape2)
library(ggplot2)
melt.OJ = melt(OJ)

ggplot(data=melt.OJ) +
   geom_boxplot(aes(x="", y=value)) +
   facet_wrap(~variable, scale="free") +
   labs(x="", y="Values")


### Figure 2: Correlation matrix plot of all quantitative variables

library(corrplot)
OJ.cor = cor(OJ[,c(2,4,5,6,7,10,11,12,13,15,16,17)])
corrplot(OJ.cor, method="square", type="upper")

########################################
### ANALYSIS USING SVM
########################################

##################################
### User function definition

### Calculates confusion matrix
print.cm = function(cm, cap) {
   table.cm = unname(cm)
   colnames(table.cm) = colnames(cm)
   rownames(table.cm) = colnames(cm)
   
   out.cm = htmlTable(table.cm,
             caption=paste(cap),
             cgroup = c("Actual"),
             rowlabel = "Predicted",
             css.cell = "width: 100px;")
   return(out.cm)
}

### Calucate performance metric from confusion matrix
my.metric = function(cm){
   accuracy = round((cm[1,1]+cm[2,2])/sum(cm),4)
   error.rate = round((cm[1,2]+cm[2,1])/sum(cm),4)
   type.1 = round(cm[2,1]/sum(cm[,1]),4)
   type.2 = round(cm[1,2]/sum(cm[,2]),4)
   sensitivity = round(cm[2,2]/sum(cm[,2]),4)
   specificity = round(cm[1,1]/sum(cm[,1]),4)
   
   metric = c(accuracy, error.rate, type.1, type.2, sensitivity, specificity)
   return(metric)
}

### Calculate area under curve
get.AUC = function(x, y) {
   dx = diff(x)
   y.mean = diff(y)/2 + y[c(1:length(y)-1)]
   
   auc = sum(dx * y.mean)
   return(auc)
}

######################################
### Training and test datasets

###
### Sample stratification
###

library(caret)

feature.str = names(OJ)
feature.groups = paste(OJ$Purchase,
                    OJ$SpecialCH,
                    OJ$SpecialMM,
                    OJ$StoreID,
                    OJ$Store7,
                    OJ$STORE)

set.seed(21)
folds.id = createFolds(feature.groups, k=4)

Purchase = OJ$Purchase

feature.ratio = function(feature, data, folds.id){
   
   n = length(folds.id)
   f = length(unique(data[, feature]))

   ratio = matrix(rep(NA, n*f), ncol=f)
   
   for(i in c(1:n)){
      
      ratio[i,] = table(data[folds.id[[i]], feature]) /
            sum(table(data[folds.id[[i]], feature]))
   }
   
   ratio = rbind(ratio)
   return(ratio)
}

get.ratio = function(feature, OJ, folds.id) {
   full.rt = table(OJ[,feature]) / sum(table(OJ[,feature]))
   rt = feature.ratio(feature, OJ, folds.id)
   ratio = rbind(full.rt, rt)
   
   rownames(ratio) = c("Fullset", "Fold1", "Fold2", "Fold3", "Fold4")
   return(ratio)
}

Purchase.ratio = get.ratio("Purchase", OJ, folds.id)
SpecialCH.ratio = get.ratio("SpecialCH", OJ, folds.id)
SpecialMM.ratio = get.ratio("SpecialMM", OJ, folds.id)
StoreID.ratio = get.ratio("StoreID", OJ, folds.id)
Store7.ratio = get.ratio("Store7", OJ, folds.id)
STORE.ratio = get.ratio("STORE", OJ, folds.id)

### Table 2: Class ratios in the full dataset and each of the fold in the test dataset

tab.ratio1 = round(cbind(Purchase.ratio,SpecialCH.ratio,SpecialMM.ratio,Store7.ratio),3)
tab.ratio2 = round(cbind(StoreID.ratio,STORE.ratio),3)

htmlTable(tab.ratio1,
          caption="Table 2: Class ratios in the full dataset and each of the fold in the test dataset",
          cgroup = c("Purchase","SpecialCH","SpecialMM","Store7"),
          n.cgroup = c(2,2,2,2),
          rowlabel = "Dataset",
          css.cell = "width: 70px;")

htmlTable(tab.ratio2,
          caption="Table 2(continued): Class ratios in the full dataset and 4-fold test dataset",
          cgroup = c("StoreID","STORE"),
          n.cgroup = c(5,5),
          rowlabel = "Dataset",
          css.cell = "width: 70px;")

######################################
### SVC

library(e1071)

n.fold = 4

ypred.train = list()
ypred.test = list()
cm.train = matrix(rep(0,n.fold), ncol=2)
cm.test = matrix(rep(0,n.fold), ncol=2)

for(i in c(1:n.fold)) {
   svmfit = svm(Purchase ~ ., data=OJ[-folds.id[[i]],], 
                kernel="linear", cost=0.01, scale=TRUE)
   
   ypred.train[[i]] = predict(svmfit)
   ypred.test[[i]] = predict(svmfit, newdata=OJ[folds.id[[i]],])
   
   cm.train = cm.train + table(Predict=ypred.train[[i]], Actual=Purchase[-folds.id[[i]]])
   cm.test = cm.test + table(Predict=ypred.test[[i]], Actual=Purchase[folds.id[[i]]])
}

train.error = (cm.train[1,2]+cm.train[2,1]) / sum(cm.train)
test.error = (cm.test[1,2]+cm.test[2,1]) / sum(cm.test)

summary(svmfit)

### Table 3: Confusion matrix based on training data <br> aggregated from all folds

print.cm(cm.train, "Table 3: Confusion matrix based on training data <br> aggregated from all folds")

### Table 4: Confusion matrix based on test data <br> aggregated from all folds

print.cm(cm.test, "Table 4: Confusion matrix based on test data <br> aggregated from all folds")

bestmod = rep(NA, n.fold)

for(i in c(1:n.fold)) {
   set.seed(10)
   tune.out = tune(svm, Purchase ~ ., data=OJ[-folds.id[[i]],], kernel="linear",
                   ranges=list(cost=c(0.01, 0.03, 0.04, 0.05, 0.06, 1)))
   bestmod[i] = tune.out$best.model$cost
}

bestmod.opt = mean(bestmod)

cat("Best 'cost' parameter for each fold:", bestmod)
cat("Selected optimal value for 'cost' parameter =", bestmod.opt)


ypred.test = list()
cm.test = matrix(rep(0,n.fold), ncol=2)
fitted = rep(NA, nrow(OJ))   # decision values
svmfit.SVC = list()

for(i in c(1:n.fold)) {
   svmfit = svm(Purchase ~ ., data=OJ[-folds.id[[i]],], 
                kernel="linear", cost=bestmod.opt, scale=TRUE)
   svmfit.SVC[[i]] = svmfit
   ypred.test[[i]] = predict(svmfit, newdata=OJ[folds.id[[i]],], decision.values=T)
   cm.test = cm.test + table(Predict=ypred.test[[i]], Actual=Purchase[folds.id[[i]]])
   
   fitted[folds.id[[i]]] = attributes(ypred.test[[i]])$decision.values
}

SVC.cm.test = cm.test
SVC.test.error = (cm.test[1,2]+cm.test[2,1]) / sum(cm.test)
SVC_test.typeI.ref = cm.test[2,1] / sum(cm.test[,1])
#SVC.test.error

### Table 5: Confusion matrix based on test data <br> aggregated from all folds using optimized SVC

print.cm(SVC.cm.test, "Table 5: Confusion matrix based on test data <br> aggregated from all folds using optimized SVC")

cutoff = seq(min(fitted)+0.01, max(fitted)-0.01, length.out=20)

sensitivity = rep(NA, 20)
typeI = rep(NA, 20)
test.error = rep(NA, 20)

for(j in c(1:20)) {
   pred.label = rep("MM", nrow(OJ))
   pred.label[which(fitted > cutoff[j])] = "CH"
   
   cm = table(pred.label, OJ$Purchase)
   
   sensitivity[j] = cm[2,2] / sum(cm[,2])
   typeI[j] = cm[2,1] / sum(cm[,1])
   test.error[j] = (cm[1,2]+cm[2,1]) / sum(cm)
}

auc.SVC = get.AUC(typeI, sensitivity)
#auc.SVC

### Figure 3: Performance characteristics of SVC on test dataset

library(ggplot2)

ggplot() +
   geom_line(aes(x=typeI, y=sensitivity), color="blue") +
   geom_point(aes(x=typeI, y=sensitivity), color="blue", size=1) +
   geom_line(aes(x=typeI, y=test.error), color="red") +
   geom_point(aes(x=typeI, y=test.error), color="red", size=1) +
   scale_y_continuous(breaks=seq(0,1,length.out=11)) +
   scale_x_continuous(breaks=seq(0,1,length.out=6)) +
   geom_vline(xintercept=SVC_test.typeI.ref) +
   annotate("text", x=SVC_test.typeI.ref+0.05, y=0.5, label="Cutoff=0") +
   annotate("text", x=0.5, y=0.9, color="blue", label="Sensitivity (ROC curve)") +
   annotate("text", x=0.7, y=0.35, color="red", label="Test error rate") +
   labs(x="Type I error", y="Value")

######################################
### SVM radial kernel

ypred.train = list()
ypred.test = list()
cm.train = matrix(rep(0,n.fold), ncol=2)
cm.test = matrix(rep(0,n.fold), ncol=2)

for(i in c(1:n.fold)) {
   svmfit = svm(Purchase ~ ., data=OJ[-folds.id[[i]],], 
                kernel="radial", cost=0.01, scale=TRUE)
   
   ypred.train[[i]] = predict(svmfit)
   ypred.test[[i]] = predict(svmfit, newdata=OJ[folds.id[[i]],])
   
   cm.train = cm.train + table(Predict=ypred.train[[i]], Actual=Purchase[-folds.id[[i]]])
   cm.test = cm.test + table(Predict=ypred.test[[i]], Actual=Purchase[folds.id[[i]]])
}

train.error = (cm.train[1,2]+cm.train[2,1]) / sum(cm.train)
test.error = (cm.test[1,2]+cm.test[2,1]) / sum(cm.test)

summary(svmfit)

### Table 6: Confusion matrix based on training data <br> aggregated from all folds

print.cm(cm.train, "Table 6: Confusion matrix based on training data <br> aggregated from all folds")

### Table 7: Confusion matrix based on test data <br> aggregated from all folds

print.cm(cm.test, "Table 7: Confusion matrix based on test data <br> aggregated from all folds")

bestmod = rep(NA, n.fold)

for(i in c(1:n.fold)) {
   set.seed(10)
   tune.out = tune(svm, Purchase ~ ., data=OJ[-folds.id[[i]],], kernel="radial",
                   ranges=list(cost=c(0.06, 0.08, 0.1, 0.3, 0.5, 1)))
   bestmod[i] = tune.out$best.model$cost
}

bestmod.opt = mean(bestmod)

cat("Best 'cost' parameter for each fold:", bestmod)
cat("Selected optimal value for 'cost' parameter=", bestmod.opt)

ypred.test = list()
cm.test = matrix(rep(0,n.fold), ncol=2)
fitted = rep(NA, nrow(OJ))   # decision values

for(i in c(1:n.fold)) {
   svmfit = svm(Purchase ~ ., data=OJ[-folds.id[[i]],], 
                kernel="radial", cost=bestmod.opt, scale=TRUE)
   ypred.test[[i]] = predict(svmfit, newdata=OJ[folds.id[[i]],], decision.values=T)
   cm.test = cm.test + table(Predict=ypred.test[[i]], Actual=Purchase[folds.id[[i]]])
   
   fitted[folds.id[[i]]] = attributes(ypred.test[[i]])$decision.values
}

SVM_rad.cm.test = cm.test
SVM_rad.test.error = (cm.test[1,2]+cm.test[2,1]) / sum(cm.test)
SVM_rad.typeI.ref = cm.test[2,1] / sum(cm.test[,1])
#SVM_rad.test.error

### Table 8: Confusion matrix based on test data <br> aggregated from all folds using optimized SVM with radial kernel

print.cm(SVM_rad.cm.test, "Table 8: Confusion matrix based on test data <br> aggregated from all folds using optimized SVM <br> with radial kernel")

cutoff = seq(min(fitted)+0.01, max(fitted)-0.01, length.out=20)

sensitivity = rep(NA, 20)
typeI = rep(NA, 20)
test.error = rep(NA, 20)

for(j in c(1:20)) {
   pred.label = rep("MM", nrow(OJ))
   pred.label[which(fitted > cutoff[j])] = "CH"
   
   cm = table(pred.label, OJ$Purchase)
   
   sensitivity[j] = cm[2,2] / sum(cm[,2])
   typeI[j] = cm[2,1] / sum(cm[,1])
   test.error[j] = (cm[1,2]+cm[2,1]) / sum(cm)
}

auc.SVM_rad = get.AUC(typeI, sensitivity)
#auc.SVM_rad

### Figure 4: Performance characteristics of SVM with radial kernal on the test dataset

library(ggplot2)

ggplot() +
   geom_line(aes(x=typeI, y=sensitivity), color="blue") +
   geom_point(aes(x=typeI, y=sensitivity), color="blue", size=1) +
   geom_line(aes(x=typeI, y=test.error), color="red") +
   geom_point(aes(x=typeI, y=test.error), color="red", size=1) +
   scale_y_continuous(breaks=seq(0,1,length.out=11)) +
   scale_x_continuous(breaks=seq(0,1,length.out=6)) +
   geom_vline(xintercept=SVM_rad.typeI.ref) +
   annotate("text", x=SVM_rad.typeI.ref+0.05, y=0.5, label="Cutoff=0") +
   annotate("text", x=0.5, y=0.9, color="blue", label="Sensitivity (ROC curve)") +
   annotate("text", x=0.7, y=0.35, color="red", label="Test error rate") +
   labs(x="Type I error", y="Value")

######################################
### SVM 2nd polynomial

ypred.train = list()
ypred.test = list()
cm.train = matrix(rep(0,n.fold), ncol=2)
cm.test = matrix(rep(0,n.fold), ncol=2)

for(i in c(1:n.fold)) {
   svmfit = svm(Purchase ~ ., data=OJ[-folds.id[[i]],], 
                kernel="polynomial", degree=2, cost=0.01, scale=TRUE)
   
   ypred.train[[i]] = predict(svmfit)
   ypred.test[[i]] = predict(svmfit, newdata=OJ[folds.id[[i]],])
   
   cm.train = cm.train + table(Predict=ypred.train[[i]], Actual=Purchase[-folds.id[[i]]])
   cm.test = cm.test + table(Predict=ypred.test[[i]], Actual=Purchase[folds.id[[i]]])
}

train.error = (cm.train[1,2]+cm.train[2,1]) / sum(cm.train)
test.error = (cm.test[1,2]+cm.test[2,1]) / sum(cm.test)

summary(svmfit)

### Table 9: Confusion matrix based on training data <br> aggregated from all folds

print.cm(cm.train, "Table 9: Confusion matrix based on training data <br> aggregated from all folds")

### Table 10: Confusion matrix based on test data <br> aggregated from all folds

print.cm(cm.test, "Table 10: Confusion matrix based on test data <br> aggregated from all folds")

bestmod = rep(NA, n.fold)

for(i in c(1:n.fold)) {
   set.seed(10)
   tune.out = tune(svm, Purchase ~ ., data=OJ[-folds.id[[i]],], 
                   kernel="polynomial", degree=2,
                   ranges=list(cost=c(1, 3, 4, 5, 7, 8, 9, 10)))
   bestmod[i] = tune.out$best.model$cost
}

bestmod.opt = mean(bestmod)

cat("Best 'cost' parameter for each fold:", bestmod)
cat("Selected optimal value for 'cost' parameter=", bestmod.opt)


ypred.test = list()
cm.test = matrix(rep(0,n.fold), ncol=2)
fitted = rep(NA, nrow(OJ))   # decision values

for(i in c(1:n.fold)) {
   svmfit = svm(Purchase ~ ., data=OJ[-folds.id[[i]],], 
                kernel="polynomial", degree=2, cost=bestmod.opt, scale=TRUE)
   ypred.test[[i]] = predict(svmfit, newdata=OJ[folds.id[[i]],], decision.values=T)
   cm.test = cm.test + table(Predict=ypred.test[[i]], Actual=Purchase[folds.id[[i]]])
   
   fitted[folds.id[[i]]] = attributes(ypred.test[[i]])$decision.values
}

SVM_poly.cm.test = cm.test
SVM_poly.test.error = (cm.test[1,2]+cm.test[2,1]) / sum(cm.test)
SVM_poly.typeI.ref = cm.test[2,1] / sum(cm.test[,1])
#SVM_poly.typeI.ref
#SVM_poly.test.error

### Table 11: Confusion matrix based on test data <br> aggregated from all folds using optimized SVM with second order polynomial kernel

print.cm(SVM_poly.cm.test, "Table 11: Confusion matrix based on test data <br> aggregated from all folds using optimized SVM <br> with second order polynomial kernel")
         
cutoff = seq(min(fitted)+0.01, max(fitted)-0.01, length.out=20)

sensitivity = rep(NA, 20)
typeI = rep(NA, 20)
test.error = rep(NA, 20)

for(j in c(1:20)) {
   pred.label = rep("MM", nrow(OJ))
   pred.label[which(fitted > cutoff[j])] = "CH"
   
   cm = table(pred.label, OJ$Purchase)
   
   sensitivity[j] = cm[2,2] / sum(cm[,2])
   typeI[j] = cm[2,1] / sum(cm[,1])
   test.error[j] = (cm[1,2]+cm[2,1]) / sum(cm)
}

auc.SVM_poly = get.AUC(typeI, sensitivity)
#auc.SVM_poly

### Figure 5: Performance characteristics of SVM with second order polynomial kernal on the test dataset

library(ggplot2)

ggplot() +
   geom_line(aes(x=typeI, y=sensitivity), color="blue") +
   geom_point(aes(x=typeI, y=sensitivity), color="blue", size=1) +
   geom_line(aes(x=typeI, y=test.error), color="red") +
   geom_point(aes(x=typeI, y=test.error), color="red", size=1) +
   scale_y_continuous(breaks=seq(0,1,length.out=11)) +
   scale_x_continuous(breaks=seq(0,1,length.out=6)) +
   geom_vline(xintercept=SVM_poly.typeI.ref) +
   annotate("text", x=SVM_poly.typeI.ref+0.05, y=0.5, label="Cutoff=0") +
   annotate("text", x=0.5, y=0.85, color="blue", label="Sensitivity (ROC curve)") +
   annotate("text", x=0.7, y=0.35, color="red", label="Test error rate") +
   labs(x="Type I error", y="Value")

###########################
### DISCUSSION
###########################

### Table 11: Performance comparsion of the SVC and SVM with radial and polynoimial kerne

auc = round(c(auc.SVC, auc.SVM_rad, auc.SVM_poly),4)

metric = rbind(my.metric(SVC.cm.test), my.metric(SVM_rad.cm.test), 
               my.metric(SVM_poly.cm.test))
colnames(metric) = c("  Accuracy  ", "Error rate", "Type I error", "Type II error", "Sensitivity", "Specificity")
rownames(metric) = c("SVC(linear)", "SVM(radial)", "SVM(poly2)")

metric = cbind(metric, auc)
colnames(metric)[7] = "AUC of ROC"

htmlTable(metric,
          caption = "Table 11: Performance comparsion of the SVC and SVM with radial and polynoimial kernel",
          css.cell="width: 100px;")

ind = c(1:1070)

f1 = ind[-folds.id[[1]]]
SV1.id = f1[svmfit.SVC[[1]]$index]

f2 = ind[-folds.id[[2]]]
SV2.id = f2[svmfit.SVC[[2]]$index]

f3 = ind[-folds.id[[3]]]
SV3.id = f3[svmfit.SVC[[3]]$index]

f4 = ind[-folds.id[[4]]]
SV4.id = f4[svmfit.SVC[[4]]$index]

SV = union(
   union(SV1.id, SV2.id),
   union(SV3.id, SV4.id)
)

SV.id = rep(0, 1070)
SV.id[SV] = 1

### Figure 6: Visualization of SVC classifier on all 1070 observations

ggplot() +
   geom_point(aes(x=OJ$LoyalCH, y=OJ$PriceDiff, color=OJ$Purchase,
                  shape=as.factor(SV.id)), size=2) +
   scale_shape_manual(values=c(17,4), labels=c("No", "Yes")) +
   labs(x="LoyalCH", y="PriceDiff", color="Actual purchase", shape="Support vector")

htmlTable(metric,
          caption = "Performance comparsion of the SVC and SVM with radial and polynoimial kernel",
          css.cell="width: 100px;")
