import sys
import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm


shortlisted = [] # will contains the input with all columns

def get_training_data():
	with open('train.csv','r') as pool:
		csv_pool = csv.reader(pool)
		for row in csv_pool:
			shortlisted.append(row)
	shortlisted.remove(shortlisted[0])

probabilities=[] #contains the array of the probabilities of the 20 ammino acids present in the string 
def calculate_probability_amino_acids_train():
	for i in shortlisted:
		templist=[]
		dic={'G': 0, 'T': 0, 'I': 0, 'F': 0, 'D': 0, 'C': 0, 'E': 0, 'S': 0, 'L': 0, 'Y': 0, 'K': 0, 'W': 0, 'N': 0, 'A': 0, 'R': 0, 'H': 0, 'V': 0, 'P': 0, 'Q': 0, 'M': 0}
		for j in i[2]:		
			dic[j]+=1
		for j in dic:
			templist.append(dic[j]/len(i[2]))
		probabilities.append(templist)

X = [] #this is also basically same as the probabilities list, but made to make the code clean
y = [] #this contains the respective +1 and -1 from the shortlisted array
sv = svm.SVC(probability=True, gamma=200, kernel='rbf')
def train_the_data():
	for i in range(len(shortlisted)):
		shortlisted[i]+=probabilities[i]
		X.append(probabilities[i])
		if shortlisted[i][1] != '-1' and shortlisted[i][1] != '1':
			shortlisted[i][1] = '-1'
			print("Invalid label at " + str(i))
		y.append(int(shortlisted[i][1]))
	sv.fit(X,y)

test_data = [] #contains the test raw data from the test.csv
sample_data = [] #contains the raw test data in from sapmle.csv
def get_test_and_sample_data():
	with open('test.csv','r') as pool:
		csv_pool = csv.reader(pool)
		for row in csv_pool:
			test_data.append(row)
	test_data.remove(test_data[0])
	with open('sample.csv','r') as pool:
		csv_pool = csv.reader(pool)
		for row in csv_pool:
			sample_data.append(row)
	sample_data.remove(sample_data[0])

probabilities_test=[]  # contains the probabilty of the 20 ammino acids of the test data
def calculate_probability_amino_acids_test():
	for i in test_data:
		templist=[]
		dic={'G': 0, 'T': 0, 'I': 0, 'F': 0, 'D': 0, 'C': 0, 'E': 0, 'S': 0, 'L': 0, 'Y': 0, 'K': 0, 'W': 0, 'N': 0, 'A': 0, 'R': 0, 'H': 0, 'V': 0, 'P': 0, 'Q': 0, 'M': 0}
		for j in i[1]:		
			dic[j]+=1
		for j in dic:
			templist.append(dic[j]/len(i[1]))
		probabilities_test.append(templist)

x_test = [] #basically probability_test only
final_sv_predicted = [] #contains the predicted +1 and -1 useing SVM

def predict_outputs():
	for i in range(len(test_data)):
		test_data[i]+=probabilities_test[i]
		x_test.append(probabilities_test[i])
		
	for i in range(len(test_data)):
		sv_predicted = sv.predict([x_test[i]])[0]

		final_sv_predicted.append(sv_predicted)

# used to create a file named SVM_output.csv and make write the final output in it
def write_to_csv():
	with open('SVM_output.csv', mode='w') as f:
		w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		w.writerow(['ID', 'Label'])
		for i in range(len(test_data)):
			w.writerow([test_data[i][0], final_sv_predicted[i]])
	print("output saved in SVM_output.csv")


if __name__ == '__main__':
	get_training_data()
	calculate_probability_amino_acids_train()
	train_the_data()
	get_test_and_sample_data()
	calculate_probability_amino_acids_test()
	predict_outputs()
	write_to_csv()