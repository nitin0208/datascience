import numpy as np 
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd 
import random
import time
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

subject="m-1"
i=0
paper_level=random.randint(1,3)
stud_prep=random.randint(1,10)
stud_iq=random.randint(1,10)
topics_covered=random.randint(1,10)

columns=['paper_level','stud_prep_time','student_iq','topics_covered','expected_result_list']

paper_level_list=[]
stud_prep_list=[]
stud_iq_list=[]
topics_covered_list=[]
expected_result_list=[]

passcount=0
failcount=0
totalcount=0

while(totalcount<1000):
	paper_level=random.randint(1,3)
	stud_prep=random.randint(1,10)
	stud_iq=random.randint(1,10)
	topics_covered=random.randint(1,10)

	if(paper_level==3):
		if(stud_iq>=6 and stud_prep>=8 and topics_covered>=6):
			result=1
		else:
			result=0
		if(stud_iq>=9 and stud_prep>=5 and topics_covered>=6):
			result=1
	if(paper_level==2):
		if(stud_iq>=6 and stud_prep>=5 and topics_covered>=5):
			result= 1
		else:
			result= 0
		if(stud_iq>=8 and stud_prep>=4 and  topics_covered>=5):
			result= 1
	if(paper_level==1):
		if(stud_iq>=3 and stud_prep>=4 and topics_covered>=3):
			result=1
		else :
			result=0
		if(stud_iq>=7 and stud_prep>=3  and topics_covered>=3):
			result=1
	if(result==1):
		passcount+=1
		totalcount+=1
	if(result==0):
		failcount+=1
		totalcount+=1
	paper_level_list.append(paper_level)
	stud_prep_list.append(stud_prep)
	stud_iq_list.append(stud_iq)
	topics_covered_list.append(topics_covered)
	expected_result_list.append(result)

print("count=",passcount,failcount,totalcount)	
paper_level=np.array(paper_level_list)
stud_prep=np.array(stud_prep_list)
stud_iq=np.array(stud_iq_list)
topics_covered=np.array(topics_covered_list)
expected_result=np.array(expected_result_list)


data={'paper_level':paper_level_list,'stud_prep':stud_prep_list,'stud_iq':stud_iq_list,'topics_covered':topics_covered_list,"expected_result":expected_result_list}
df=pd.DataFrame(data)
print(df)

X = np.array(df.drop(['expected_result'],1))
y = np.array(df['expected_result'])

# X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
# x_train=df([[[['paper_level','stud_prep','stud_iq','topics_covered']]]]).reshape(-1,1)

# y_train=df["expected_result"]
clf = neighbors.KNeighborsClassifier()
clf.fit(X,y)

# accurary = clf.score(X_test,y_test)

# print (accurary)
print (X,y)
test_set = np.array([[1,7,9,8]])
prediction = clf.predict(test_set)

print ('\n',prediction)


