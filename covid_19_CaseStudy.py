#Jumana Rahman
import numpy as np
import pandas as pd
from turtle import hideturtle
from scipy.stats import norm
from statistics import mode
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind, ttest_rel, ttest_ind_from_stats

from scipy.stats import norm
from statistics import mode
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
from turtle import hideturtle
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind, ttest_rel, ttest_ind_from_stats
from scipy.stats import norm
from statistics import mode
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#################################################################################################################
#load the .csv data
reviewDataOrig = pd.read_csv(r"C:/Users/19172/Desktop/covid_student_responses.csv")
#print(reviewDataOrig)

#drop any columns that are unneccessary
#use below for model training and prediction
reviewData = reviewDataOrig.drop(columns = ["ID", "Region of residence", "Time spent on Online Class", "Rating of Online Class experience",
                                            "Medium for online class", "Time spent on TV", "Number of meals per day",
                                            "Change in your weight", "Health issue during lockdown", "Time utilized",
                                            "Do you find yourself more connected with your family, close friends , relatives  ?", "What you miss the most", "Stress busters"], axis = 1)
#print(reviewData)

#use below for study on dataset statistics to find patterns, connections, and relations
studentLifeData = reviewDataOrig.drop(columns = ["ID", "Region of residence","Age of Subject" ,"Time spent on Online Class", "Rating of Online Class experience",
                                            "Medium for online class", "Time spent on TV", "Number of meals per day",
                                            "Change in your weight", "Health issue during lockdown","Stress busters", "Time utilized",
                                            "Do you find yourself more connected with your family, close friends , relatives  ?",
                                            "What you miss the most"], axis = 1)
print(studentLifeData)
print("------------------------------------------------")

#check if have any null values, if do, drop null values
print(reviewData.isnull().any())
print("------------------------------------------------")
print(studentLifeData.isnull().any())
print("------------------------------------------------")

#################################################################################################################
 #EXPLORE DATA STATISTICALLY

#1. Find patterns amongst the data you chose to explore 
#(Includes the  dimensions of the dataset, the central tendency, standard deviation, and 5- number summary and normal distribution)
    #1a. Find patterns in student's time spent on social media
    
#plot the histogram below for self study
print("Summarizing statistical data for self study time:")
plt.hist(studentLifeData['Time spent on self study'])
plt.xlabel('Time spent on self study')
plt.ylabel('count')
plt.title("Student's time spent on self study during Covid-19")
plt.show() 
#mean, median, standard deviation for self study time
print(studentLifeData['Time spent on self study'].astype(float).describe())
#calculate mode
print('The modes for the time spent studying are ', mode(studentLifeData['Time spent on self study']))

print("--------------------------------------------------------------------")
    #1b. Find patterns in student's time spent on fitness
print("Summarizing statistical data for fitness time:")
plt.hist(studentLifeData['Time spent on fitness'])
plt.xlabel('Time Spent on Fitness')
plt.ylabel('Count')
plt.title("Student's Time Spent on Fitness During Covid-19")
plt.show() 
#mean, median, standard deviation for fitness time
print(studentLifeData['Time spent on fitness'].astype(float).describe())
#calculate mode
print('The modes for the time spent on fitness are ', mode(studentLifeData['Time spent on fitness']))

print("--------------------------------------------------------------------")
    #1c. Find patterns in student's time spent on sleep
print("Summarizing statistical data for sleep time:")
plt.hist(studentLifeData['Time spent on sleep'])
plt.xlabel('Time Spent on Sleep')
plt.ylabel('Count')
plt.title("Student's Time Spent on Sleep During Covid-19")
plt.show() 
#mean, median, standard deviation for sleep time
print(studentLifeData['Time spent on sleep'].astype(float).describe())
#calculate mode
print('The modes for the time spent on sleep are ', mode(studentLifeData['Time spent on sleep']))

print("--------------------------------------------------------------------")
    #1d. Find patterns in student's time spent on social media
print("Summarizing statistical data for social media time:")
plt.hist(studentLifeData['Time spent on social media'])
plt.xlabel('Time Spent on Social Media')
plt.ylabel('Count')
plt.title("Student's Time Spent on Social Media During Covid-19")
plt.show() 
#mean, median, standard deviation for sleep time
print(studentLifeData['Time spent on social media'].astype(float).describe())
#calculate mode
print('The modes for the time spent on social media are ', mode(studentLifeData['Time spent on social media']))

print("--------------------------------------------------------------------")
#----------------------------------------------------------------------------------------------------------------

#2. Find Connection amongst the data you chose to explore using correlation
    #2a. Explore the connection between time spent on social media and time spent on fitness
print("Correlation data between time spent on social media and time spent on fitness: ")
socialVersusFitness = studentLifeData[['Time spent on social media', 'Time spent on fitness']]
print(socialVersusFitness)

x = socialVersusFitness['Time spent on social media'].astype(float)  #made x be time social media, seeing the affect social media may have on fitness time
                                      
y = socialVersusFitness['Time spent on fitness'].astype(float) 
plt.scatter(x,y)
plt.xlabel("Time spent on social media")
plt.ylabel("Time spent on fitness")
plt.title("Student's Time spent on Social Media in Comparison to Fitness")
plt.show()

#find r
fitnessr = y.corr(x)             # gave -0.04285516491983569  

print("The Pearson's Correlation coeficient, r is ", fitnessr, "\n")

print("------------------------------------------------")
    #2b. Explore the connection between time spent on social media and time spent on self study
print("Correlation data between time spent on social media and time spent on self study: ")
socialVersusStudy = studentLifeData[['Time spent on social media', 'Time spent on self study']]
print(socialVersusStudy)

x = socialVersusStudy['Time spent on social media'].astype(float)  #made x be time social media, seeing the affect social media may have on study time
                                       
y = socialVersusStudy['Time spent on self study'].astype(float) 
plt.scatter(x,y)
plt.xlabel("Time spent on social media")
plt.ylabel("Time spent on self study")
plt.title("Student's Time spent on Social Media in Comparison to Self Study")
plt.show()

#find r
studyr = y.corr(x)             # gave -0.1626125113743757

print("The Pearson's Correlation coeficient, r is ", studyr, "\n")

print("------------------------------------------------")
    #2c. Explore the connection between time spent on social media and time spent on sleep
print("Correlation data between time spent on social media and time spent on sleep: ")
socialVersusSleep = studentLifeData[['Time spent on social media', 'Time spent on sleep']]
print(socialVersusSleep)

x = socialVersusSleep['Time spent on social media'].astype(float)  #made x be time social media, seeing the affect social media may have on sleep time
                                       
y = socialVersusSleep['Time spent on sleep'].astype(float) 
plt.scatter(x,y)
plt.xlabel("Time spent on social media")
plt.ylabel("Time spent on self sleep")
plt.title("Student's Time spent on Social Media in Comparison to Sleep")
plt.show()

#find r
sleepr = y.corr(x)             # gave    

print("The Pearson's Correlation coeficient, r is ", sleepr, "\n")

print("------------------------------------------------")
#----------------------------------------------------------------------------------------------------------------
#3a. Normal Distribution for self study
print("Doing Normal Distribution for time spent on self study")
study_mean_100 = []
for i in range (10000):
    studysample = np.random.choice(studentLifeData['Time spent on self study'], 100).astype('float').mean()
    study_mean_100.append(studysample)
    
overall_study_mean = np.mean(study_mean_100)
print("Sampling mean for self study: ", overall_study_mean)      
# thus the sampling mean is 2.9097146

#show the normally distributed sampling distribution below
plt.hist(study_mean_100)
plt.xlabel("Time students spent on self study")
plt.ylabel("count")
plt.title("Overall Time Students Spent on Self Study During Covid-19")
plt.show()

print("------------------------------------------------")
#3b. Normal Distribution for fitness
print("Doing Normal Distribution for time spent on fitness")
fitness_mean_100 = []
for i in range (10000):
    fitnesssample = np.random.choice(studentLifeData['Time spent on fitness'], 100).astype('float').mean()
    fitness_mean_100.append(fitnesssample)
    
overall_fitness_mean = np.mean(fitness_mean_100)
print("Sampling mean for Fitness: ", overall_fitness_mean)      
# thus the sampling mean is 0.7665098

#show the normally distributed sampling distribution below
plt.hist(fitness_mean_100)
plt.xlabel("Time students spent on fitness")
plt.ylabel("Count")
plt.title("Overall Time Students Spent on Fitness During Covid-19")
plt.show()

print("------------------------------------------------")
#3c. Normal Distribution for sleep
print("Doing Normal Distribution for time spent on sleep")
sleep_mean_100 = []
for i in range (10000):
    sleepsample = np.random.choice(studentLifeData['Time spent on sleep'], 100).astype('float').mean()
    sleep_mean_100.append(sleepsample)
    
overall_sleep_mean = np.mean(sleep_mean_100)
print("Sampling mean for Sleep: ", overall_sleep_mean)      
# thus the sampling mean is 7.8699322

#show the normally distributed sampling distribution below
plt.hist(sleep_mean_100)
plt.xlabel("Time students spent on sleep")
plt.ylabel("Count")
plt.title("Overall Time Students Spent on Sleep During Covid-19")
plt.show()

print("------------------------------------------------")
#3d. Normal Distribution for social media
print("Doing Normal Distribution for time spent on social media")
socialmedia_mean_100 = []
for i in range (10000):
    socialmediasample = np.random.choice(studentLifeData['Time spent on social media'], 100).astype('float').mean()
    socialmedia_mean_100.append(socialmediasample)
    
overall_socialmedia_mean = np.mean(socialmedia_mean_100)
print("Sampling mean for Social Media: ", overall_socialmedia_mean)      
# thus the sampling mean is 2.3644672000000004

#show the normally distributed sampling distribution below
plt.hist(socialmedia_mean_100)
plt.xlabel("Time students spent on social media")
plt.ylabel("Count")
plt.title("Overall Time Students Spent on Social Media During Covid-19")
plt.show()

print("------------------------------------------------") 
##########################################################################################
#3. Questions to find relationships in the dataset
#question1 done
print("Question 1:Are the times spent on self study where time spent on social media is high(greater than or equal to 2.0) different than the times spent on self study where the time spent of social media is low(less than 2.0)?\n")


studyonsocial = studentLifeData[['Time spent on self study', 'Time spent on social media']].astype(float)
studyonsocial.rename(columns = {'Time spent on self study':'studytime', 'Time spent on social media':'mediatime'}, inplace = True)
social_high = studyonsocial.query('mediatime>=2.0')
social_low = studyonsocial.query('mediatime<2.0')

t, p = ttest_ind(a = social_high['studytime'],b = social_low['studytime'], equal_var = False )
print("The p value for this case is: ", p)
if p <= 0.05:
   print('Conclusion: Reject H0 and accept HA to say that the time students spent on self study is different for when the time on social media is high compared to when the time on social media is low.\n')
else:
    print('Conclusion: Accept H0 and say that the time students spent on self study is different for when the time on social media is high compared to when the time on social media is low.')
    

print("------------------------------------------------")
#---------------------------------------------------------------------------------------------------------------------------------
#question2 done
print("------------------------------------------------")
print("Question 2: Amongst students who chose Instagram as the medium for social media platforms, the time spent on sleep is lower where the time spent on social media is high compared to the times where social media time is low. \n")

instagramData= studentLifeData[['Time spent on social media', 'Time spent on sleep', 'Prefered social media platform']]
instagramData.rename(columns = {'Time spent on social media':'mediatime', 'Time spent on sleep':'sleeptime', 'Prefered social media platform':'chosenmedia'}, inplace = True)

choseninsta = instagramData.query('chosenmedia == "Instagram"')

social_high = choseninsta.query('mediatime>=2.0')
social_low = choseninsta.query('mediatime<2.0')

#print("rows in highsoc ",social_high.shape[0])  # face an issue because not the same size
#print("rows in lowsoc ",social_low.shape[0])

social_high = social_high[:-182]   #have to trim so they're the same size

t, p = ttest_rel(a = social_low['sleeptime'],b = social_high['sleeptime'])

print("The p value for this case is: ", p)
if p <= 0.05:
    print('Conclusion: Reject H0 and accept HA to say that amongst students that chose Instagram as their preferred medium, the time spent on sleep is the lower for students that spent high amounts of time on social media compared to the students that spent low amount of time on social media.  \n')
else:
    print('Conclusion: Accept H0 and say that amongst students that chose Instagram as their preferred medium, the time spent on sleep is the same for students that spent high amounts of time on social media as the students that spent low amount of time on social media.')
 







#question 3 affect instagram has on study time for students that spend a lot of time on social media done
print("------------------------------------------------")
print("Question 3: Amongst students who chose Instagram as the medium for social media platforms, is the time spent on self study is lower where the time spent on social media is high compared to the times where social media time is low.  \n")


instagramstudyData= studentLifeData[['Time spent on social media', 'Time spent on self study', 'Prefered social media platform']]
instagramstudyData.rename(columns = {'Time spent on social media':'mediatime', 'Time spent on self study':'selfstudytime', 'Prefered social media platform':'chosenmedia'}, inplace = True)

choseninstastudy = instagramstudyData.query('chosenmedia == "Instagram"')

social_high = choseninstastudy.query('mediatime>=2.0')
social_low = choseninstastudy.query('mediatime<2.0')

#print("rows in highsoc ",social_high.shape[0])  # face an issue because not the same size
#print("rows in lowsoc ",social_low.shape[0])

social_high = social_high[:-182]   #have to trim so they're the same size

t, p = ttest_rel(a = social_low['selfstudytime'],b = social_high['selfstudytime'])

print("The p value for this case is: ", p)
if p <= 0.05:
    print('Conclusion: Reject H0 and accept HA to say that amongst students who chose Instagram as the medium for social media platforms, the time spent on self study is lower where the time spent on social media is high compared to the times where social media time is low.\n')
else:
    print('Conclusion: Accept H0 and say that amongst students who chose Instagram as the medium for social media platforms, the time spent on self study is the same where the time spent on social media is high compared to the times where social media time is low.')



#question 4  affect linkedIn has on study time for students that spend a lot of time on social media
print("------------------------------------------------")
print("Question 4: Amongst students who chose LinkedIn as the medium for social media platforms, is the time spent on self study is lower where the time spent on social media is high compared to the times where social media time is low.  \n")


linkedInstudyData= studentLifeData[['Time spent on social media', 'Time spent on self study', 'Prefered social media platform']]
linkedInstudyData.rename(columns = {'Time spent on social media':'mediatime', 'Time spent on self study':'selfstudytime', 'Prefered social media platform':'chosenmedia'}, inplace = True)

chosenlinkedInstudy = linkedInstudyData.query('chosenmedia == "Linkedin"')

social_high = chosenlinkedInstudy.query('mediatime>=2.0')
social_low = chosenlinkedInstudy.query('mediatime<2.0')

#print("rows in highsoc ",social_high.shape[0])  # face an issue because not the same size
#print("rows in lowsoc ",social_low.shape[0])

social_high = social_high[:-15]   #have to trim so they're the same size

t, p = ttest_rel(a = social_low['selfstudytime'],b = social_high['selfstudytime'])

print("The p value for this case is: ", p)
if p <= 0.05:
    print('Conclusion: Reject H0 and accept HA to say that a\n')
else:
    print('Conclusion: Accept H0 and say that amongst students who chose Instagram as the medium for social media platforms, the time spent on self study is the same where the time spent on social media is high compared to the times where social media time is low.')

################################################################################################

#question3a calculating independence between social media and fitness
print("------------------------------------------------")
print("Testing for independence between students social media time and fitness time")


fitnessSocial = studentLifeData[['Time spent on fitness', 'Time spent on social media']].astype(float)

fitnessSocial.rename(columns = {'Time spent on fitness':'fitnesstime', 'Time spent on social media':'mediatime'}, inplace = True)

hifithisoc = fitnessSocial.query('mediatime >= 2.0 & fitnesstime >= 1.0').shape[0]
hifitlosoc = fitnessSocial.query('mediatime >= 2.0 & fitnesstime < 1.0').shape[0]
lofithisoc = fitnessSocial.query('mediatime < 2.0 & fitnesstime >= 1.0').shape[0]
lofitlosoc = fitnessSocial.query('mediatime < 2.0 & fitnesstime < 1.0').shape[0]

u = np.array([[hifithisoc, hifitlosoc],[lofithisoc, lofitlosoc]])
style = pd.DataFrame(u,index = ['Media Time greater than or equal to 2', 'Media Time less than 2'], columns = ['Fitness Time greater than or equal to 2', 'Media Time less than 2'])

print(style)  #successfully made table as shown below


stat,p,dof,expected = chi2_contingency(style)
alpha = 0.5
print("p value is "+ str(p))
if p <= alpha:
    print('Conclusion: The two variables are dependent(reject H0)')
else:
    print('Conclusion: The two variables are independent of each other (fail to reject H0)')
    
''' Result:
Testing for independence between students social media time and fitness time
                                       Fitness Time greater than or equal to 2  Media Time less than 2
Media Time greater than or equal to 2                                      424                     317
Media Time less than 2                                                     273                     168
p value is 0.12789738176130594
Conclusion: The two variables are dependent(reject H0)'''
#------------------------------------------------------------------------------------------------------------------
#question3b calculating independence between social media and self study time done

print("------------------------------------------------")
print("Testing for independence between students social media time and self study time")

studySocial = studentLifeData[['Time spent on social media', 'Time spent on self study']].astype(float)

studySocial.rename(columns = {'Time spent on social media':'mediatime','Time spent on self study':'stutime'}, inplace = True)

histuhisoc = studySocial.query('mediatime >= 2.0 & stutime >= 2.0').shape[0]
histulosoc = studySocial.query('mediatime >= 2.0 & stutime < 2.0').shape[0]
lostuhisoc = studySocial.query('mediatime < 2.0 & stutime >= 2.0').shape[0]
lostulosoc = studySocial.query('mediatime < 2.0 & stutime < 2.0').shape[0]

u = np.array([[histuhisoc, histulosoc],[lostuhisoc, lostulosoc]])
style = pd.DataFrame(u,index = ['Media Time greater than or equal to 2', 'Media Time less than 2'], columns = ['Fitness Time greater than or equal to 2', 'Media Time less than 2'])

print(style)  #successfully made table as shown below

stat,p,dof,expected = chi2_contingency(style)
alpha = 0.5
print("p value is "+ str(p))
if p <= alpha:
    print('Conclusion: The two variables are dependent(reject H0)')
else:
    print('Conclusion: The two variables are independent of each other (fail to reject H0)')

''' Testing for independence between students social media time and self study time
                                       Fitness Time greater than or equal to 2  Media Time less than 2
Media Time greater than or equal to 2                                      555                     186
Media Time less than 2                                                     354                      87
p value is 0.040505601914242344
Conclusion: The two variables are dependent(reject H0) '''
#------------------------------------------------------------------------------------------------------------------
#question3c calculating independence between social media and self study time

print("------------------------------------------------")
print("Testing for independence between students social media time and sleep time")

sleepSocial = studentLifeData[['Time spent on social media', 'Time spent on sleep']].astype(float)

sleepSocial.rename(columns = {'Time spent on social media':'mediatime','Time spent on sleep':'sleeptime'}, inplace = True)

hislehisoc = sleepSocial.query('mediatime >= 2.0 & sleeptime >= 8.0').shape[0]
hislelosoc = sleepSocial.query('mediatime >= 2.0 & sleeptime < 8.0').shape[0]
loslehisoc = sleepSocial.query('mediatime < 2.0 & sleeptime >= 8.0').shape[0]
loslelosoc = sleepSocial.query('mediatime < 2.0 & sleeptime < 8.0').shape[0]

u = np.array([[hislehisoc, hislelosoc],[loslehisoc, loslelosoc]])
style = pd.DataFrame(u,index = ['Media Time greater than or equal to 2', 'Media Time less than 2'], columns = ['Fitness Time greater than or equal to 2', 'Media Time less than 2'])

print(style)  #successfully made table as shown below

stat,p,dof,expected = chi2_contingency(style)
alpha = 0.5
print("p value is "+ str(p))
if p <= alpha:
    print('Conclusion: The two variables are dependent(reject H0)')
else:
    print('Conclusion: The two variables are independent of each other (fail to reject H0)')

''' Testing for independence between students social media time and self study time
                                       Fitness Time greater than or equal to 2  Media Time less than 2
Media Time greater than or equal to 2                                      555                     186
Media Time less than 2                                                     354                      87
p value is 0.040505601914242344
Conclusion: The two variables are dependent(reject H0) '''

print("______________________________________________________________________________")
print("PART 2")

#################################################################################################################
#4.Build a predictive model by using Naïve bayes algorithm to predict whether the next student to write a review 
#based on age, time spent studying, fitness, sleep, social media, stress buster, what they miss most
#Goal: Predict if the next student to write a review choose Instagram to be their preferred social media platform along with get insight into how this student will spend his/her day.
le = LabelEncoder()
reviewData['Prefered social media platform']= le.fit_transform(reviewData['Prefered social media platform'])
print(reviewData)
x = reviewData.iloc[:,[0,1,2,3,4]].values  #independent variable all other columns
print(x)
y = reviewData.iloc[:,4].values      #target variable preferred platform type
y = y.astype('int')
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
print(x_train[:5]) #print first five rows of trained data

print("------------------------------------------------")

#did below to get a better idea of predict more information about the student
mean_age = x_train[:,0].mean()
std_age = x_train[:,0].std()

mean_studytime = x_train[:,1].mean()
std_studytime = x_train[:,1].std()

mean_fitnesstime = x_train[:,2].mean()
std_fitnesstime = x_train[:,2].std()

mean_sleeptime = x_train[:,3].mean()
std_sleeptime = x_train[:,3].std()

mean_mediatime = x_train[:,4].mean()
std_mediatime = x_train[:,4].std()

#Standardization
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
#print(x_train[0,0])   # we get same result as line 52: -1.0

x_test = sc.transform(x_test) #to perform centering and scaling for xtest

#Now to import our Gaussian Model
gnb = GaussianNB()
gnb.fit(x_train,y_train) #train the model using the training set

y_predict = gnb.predict(x_test)
print(y_predict)


#Perform calculations to predict more information on the student


#____________________________________
#evaluate the model's output
print("------------------------------------------------")
#calculate accuracy
print("The accuracy of the Gaussian Model is: ",accuracy_score(y_test, y_predict))
print("------------------------------------------------")

#Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix: \n")
print(cm)
print("------------------------------------------------")

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity: ", sensitivity)
print("------------------------------------------------")

specificity = cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity: ", specificity)
print("------------------------------------------------")

#Prediction
print(gnb.predict([[20,2, 1, 5, 2]]))

reviewData['Prefered social media platform']= le.fit_transform(reviewData['Prefered social media platform'])

#calculations to predict more information on the customer
age_p = (-1)*std_age +mean_age
print("Age: ", age_p)

studytime_p = (-1)*std_studytime +mean_studytime
print("Time spent on self study: ", studytime_p)

fitnesstime_p = (-1)*std_fitnesstime +mean_fitnesstime
print("Time spent on fitness: ", fitnesstime_p)

sleeptime_p = (-1)*std_sleeptime +mean_sleeptime
print("Time spent on sleep: ", sleeptime_p)

mediatime_p = (-1)*std_mediatime +mean_mediatime
print("Time spent on social media: ", mediatime_p)


