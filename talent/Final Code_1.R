install.packages("eeptools")
library(gains)
library(dplyr)
library(gains)
library(irr)
library(caret)
library(readxl)
library(lubridate)
library(eeptools)
install.packages('e1071', dependencies=TRUE)


file1<-read_excel("C:\\Madhan\\Consulting\\My_AI_works\\My_AI_works\\Data.xlsx",sheet = 1)
names(file1)

file1%>%mutate(Target=ifelse(OverallStatus=="Joined",1,0))->file1
table(file1$Target)
#creating dummy variables
table(file1$Hiretype)
file1$dum_hiretype <- NULL
file1$dum_hiretype[file1$Hiretype=='New Employee']=1
file1$dum_hiretype[file1$Hiretype=='Re hire']=2
file1$dum_hiretype[file1$Hiretype=='Transfer']=3
table(file1$dum_hiretype)


table(file1$RecruiterGPN)
file1$dum_rec<-NULL
file1$dum_rec[file1$RecruiterGPN=="XE020M38015"]=1
file1$dum_rec[file1$RecruiterGPN=="XE020M42382"]=2
file1$dum_rec[file1$RecruiterGPN=="XE020M32824"]=3
file1$dum_rec[file1$RecruiterGPN=="XE020M39949"]=4
file1$dum_rec[file1$RecruiterGPN=="XE020M24041"]=5
file1$dum_rec[file1$RecruiterGPN=="XE020M36003"]=6
file1$dum_rec[file1$RecruiterGPN=="XE020M51545"]=7
file1$dum_rec[file1$RecruiterGPN=="XE020M63277"]=8
file1$dum_rec[file1$RecruiterGPN=="XE020M32654"]=9
file1$dum_rec[file1$RecruiterGPN=="XE020M22317"]=10
file1$dum_rec[file1$RecruiterGPN=="XE021306642"]=11
file1$dum_rec[file1$RecruiterGPN=="XE020M17867"]=12
file1$dum_rec[file1$RecruiterGPN=="XE020M24506"]=13
file1$dum_rec[file1$RecruiterGPN=="XE020M24506"|
                file1$RecruiterGPN=="XE020M00201"|
                file1$RecruiterGPN=="XE021305129"|file1$RecruiterGPN=="XE020M63244"|
                file1$RecruiterGPN=="XE021301732"|file1$RecruiterGPN=="XE021306728"
              |file1$RecruiterGPN=="IN010M41415"|file1$RecruiterGPN=="XE021305189"
              |file1$RecruiterGPN=="XE020M20654"]=14
table(file1$dum_rec)

table(file1$Title)
file1$dum_title <- NULL
file1$dum_title[file1$Title=="Mr"]=1
file1$dum_title[file1$Title=="Ms"]=2
table(file1$dum_title)
View(file1$Title) #Title got encoded as 0 and 1

summary(file1)
file1$DateOfBirth <- as.POSIXct(file1$DateOfBirth, format = "%d-%m-%Y")
file1$date <- today()
file1$date1 <- as.POSIXct(file1$date, format = "%d-%m-%Y")
file1$age <- round((file1$date1-file1$DateOfBirth)/365.25)
View(file1$age)

table(file1$`Original State`)
file1$dum_State <- NULL
file1$dum_State[file1$`Original State`=='Karnataka']=1
file1$dum_State[file1$`Original State`=='Tamil Nadu']=2
file1$dum_State[file1$`Original State`=='Kerala']=3
file1$dum_State[file1$`Original State`=='Andhra Pradesh']=4
file1$dum_State[file1$`Original State`=='New Delhi']=6
file1$dum_State[file1$`Original State`=='West Bengal']=5
file1$dum_State[file1$`Original State`=='Haryana']=7
file1$dum_State[file1$`Original State`=='Maharashtra']=8
file1$dum_State[file1$`Original State`=='Assam'|file1$`Original State`=='Chhattisgarh'|
                  file1$`Original State`=='Gujarat'|file1$`Original State`=='HARYANA'|
                  file1$`Original State`=='Himachal Pradesh'|
                  file1$`Original State`=='Jharkhand'|file1$`Original State`=='Madhya Pradesh'|
                  file1$`Original State`=='Maharashtra'|file1$`Original State`=='Orissa'|
                  file1$`Original State`=='Uttar Pradesh' | file1$`Original State`=='Uttarakhand'
                |file1$`Original State`=='Telangana'|file1$`Original State`=='Rajasthan'|
                  file1$`Original State`=='Punjab'|file1$`Original State`=='Chhattisgarh'|
                  file1$`Original State`=='England'|file1$`Original State`=='Australia'
                  |file1$`Original State`=='Georgia'|file1$`Original State`=='Sharjah'
                |file1$`Original State`=='Jammu and Kashmir']=9
table(file1$dum_State)


table(file1$Source)
file1$dum_Source <- NULL
file1$dum_Source[file1$Source=='EMPLOYEE REFERRAL']=1
file1$dum_Source[file1$Source=='CONSULTANCY']=2
file1$dum_Source[file1$Source=='PORTAL']=3
file1$dum_Source[file1$Source=='CAMPUS'| file1$Source=='OFF CAMPUS']=4
file1$dum_Source[file1$Source=='GDS LEAP'|file1$Source=='CWR TO FTE CONVERSION'
                 |file1$Source=='ALTERNATE SOURCE'|file1$Source=='WALK-IN'
                 |file1$Source=='TRANSFER'|file1$Source=='OTHER TRANSFER'
                 |file1$Source=='ADVERTISEMENT']=5
table(file1$dum_Source)

table(file1$Department)
file1$dum_Department <- NULL
file1$dum_Department[file1$Department=='GCR']=1
file1$dum_Department[file1$Department=='PAS']=2
file1$dum_Department[file1$Department=='Tax Chat']=3
file1$dum_Department[file1$Department=='Tax - TTT']=4
file1$dum_Department[file1$Department=='Indirect']=5
file1$dum_Department[file1$Department=='Transfer Pricing']=6
file1$dum_Department[file1$Department=='GDS']=7
file1$dum_Department[file1$Department=='MENA'|file1$Department=='Tax Experience Management Team'
                     |file1$Department=='PCS'|file1$Department=='Tax ACR'|
                       file1$Department=='Tax Law Services'|file1$Department=='Payroll'|
                       file1$Department=='Tax Operation Support'|file1$Department=='ITS EMS']=8

table(file1$dum_Department)

table(file1$'Joining Center')
file1$dum_Joining_Center <- NULL
file1$dum_Joining_Center[file1$'Joining Center'=="Bangalore"]=1
file1$dum_Joining_Center[file1$'Joining Center'=="Gurgaon"]=2
file1$dum_Joining_Center[file1$'Joining Center'=='Chennai']= 3
file1$dum_Joining_Center[file1$'Joining Center'=='Kochi'
                         |file1$'Joining Center'=='Trivandrum']=4
table(file1$dum_Joining_Center)


table(file1$Rank)
file1$dum_Rank <- NULL
file1$dum_Rank[file1$Rank==44]=1
file1$dum_Rank[file1$Rank==42]=2
file1$dum_Rank[file1$Rank==4]=3
file1$dum_Rank[file1$Rank==32]=4
table(file1$dum_Rank)

file1$dummy_IsRotationalShift <- ifelse((file1$IsRotationalShift== "False"),1,0)
table(file1$dummy_IsRotationalShift)


table(file1$Qualification1)
file1$dum_Qual <- NULL
file1$dum_Qual[file1$Qualification1=="Post Grad"]=1
file1$dum_Qual[file1$Qualification1=="Grad"]=2
file1$dum_Qual[file1$Qualification1=="MBA"]=3
file1$dum_Qual[file1$Qualification1=="CA"]=4
table(file1$dum_Qual)


file1$ExpectedJoiningDate <- as.POSIXct(file1$ExpectedJoiningDate, format = "%d-%m-%Y")
file1$joiningmonth <-month(as.POSIXlt(file1$ExpectedJoiningDate, format="%d/%m/%Y"))
file1$IsNoticePeriodReimbursementProvided<-as.factor(file1$IsNoticePeriodReimbursementProvided)
file1$IsNoticePeriodReimbursementProvided[is.na(file1$IsNoticePeriodReimbursementProvided)] = 0

summary(file1)
names(file1)


set.seed(200)
index<-sample(nrow(file1),0.70*nrow(file1),replace = F)
train<-file1[index,]
test<-file1[-index,]
table(file1$Target)
table(file1$PinCode)

View(train)

mod<-glm(Target~ PinCode + IT + NoticePeriod +`Big four`+ DAndI + `hike%` +
           JoiningBonus + GoodsTransportationTravelEligibility + Stipend +
           Experienced + IsNoticePeriodReimbursementProvided +
           dum_hiretype + dum_rec + Age + dum_State + Title +
           dum_Source + dum_Department + dum_Joining_Center +
           dum_Rank + dummy_IsRotationalShift + dum_Qual + joiningmonth ,data = train , family = "binomial",na.omit(train))

summary(mod)

names(train)

mod1<-glm(Target~ PinCode + IT + NoticePeriod +`Big four`+ `hike%` +
           JoiningBonus + GoodsTransportationTravelEligibility + Stipend +
           Experienced + IsNoticePeriodReimbursementProvided +
           dum_hiretype + dum_rec + Age + dum_State +
           dum_Source + dum_Department + dum_Joining_Center +
           dum_Rank + dummy_IsRotationalShift + dum_Qual + joiningmonth ,data = train , family = "binomial")

summary(mod1)


mod2<-glm(Target~ IT + NoticePeriod +`Big four`+ `hike%` +
            JoiningBonus + GoodsTransportationTravelEligibility + Stipend +
            Experienced + IsNoticePeriodReimbursementProvided +
            dum_hiretype + dum_rec + dum_title + Age + dum_State +
            dum_Source + dum_Department + dum_Joining_Center +
            dum_Rank + dummy_IsRotationalShift + dum_Qual + joiningmonth ,data = train , family = "binomial",na.action = na.exclude(train))

summary(mod2)

#Removed dummy_IsRotationalShift due to it has only 0 category rows
mod3<-glm(Target~ IT + NoticePeriod +`Big four`+ `hike%` +
              JoiningBonus +Experienced + dum_hiretype  + Title +
            dum_Source  + dum_Rank + dum_Qual  ,data = train , family = "binomial",na.action = na.exclude(train))

write.csv(train,'C:\\Madhan\\Consulting\\My_AI_works\\My_AI_works\\Rtrain.csv')
write.csv(test,'C:\\Madhan\\Consulting\\My_AI_works\\My_AI_works\\Rtest.csv')


summary(mod3)
View(train)


file3<-read_excel("C:/Users/astha.verma/Desktop/Talent forecasting/current_file1308.xlsx",sheet = 1)
names(file3)

predictions<-cbind(file3,pred = predict(mod3,type = "response",newdata = file3))
write.csv(predictions,file = "C:\Users\Nisha.Nath\AppData\Local\Programs\Python\Python37-32\Scripts\Python files\predictions.csv", row.names = F)
mean(pred)
table(file1$Target)/nrow(file1)
pred<-ifelse(pred>=0.5569177,1,0)
table(pred)
kappa2(data.frame(test$Target,pred))
confusionMatrix(pred,test$Target,positive = "1")
class(pred)
class(test$Target)
pred<-as.factor(pred)
test$Target<-as.factor(test$Target)





