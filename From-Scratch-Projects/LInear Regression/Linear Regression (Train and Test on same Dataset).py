x = input("Enter all indenepent values seperated by spaces = ")
list_of_independent = x.split(' ')
average_x = 0
independent = []
total_independent = len(list_of_independent)
for i in range(len(list_of_independent)):
    average_x += float(list_of_independent[i])
    independent.append(float(list_of_independent[i]))
y = input("Enter all dependent values seperated by spaces = ")
list_of_dependent = y.split(' ')
average_y = 0
dependent = []
total_dependent = len(list_of_dependent)
for i in range(len(list_of_dependent)):
    average_y += float(list_of_dependent[i])
    dependent.append(float(list_of_dependent[i]))
print("Total Independent values = ",total_independent)
print("Total Dependent values  = ",total_dependent)
average_x = average_x/len(independent)
average_y = average_y/len(dependent)
a = 0.0
b = 0.0
for i in range(len(list_of_independent)):
    a += (independent[i]-average_x)*(dependent[i]-average_y)
    b += (independent[i]-average_x)**2
theta1 = a/b
theta0 = average_y - (theta1*average_x)
check = input("Do you also want to evaluate your model on the same dataset? Enter (y) for yes or (n) for no: ")
if (check == 'y'):
    per = int(input("What percentage do you want to split for testing? Enter Value eg. 20 for 20%: "))
    test_value = (total_independent*per)//100
    independent_test = []
    dependent_test = []
    for i in range(test_value):
        independent_test.append(independent[i])
        dependent_test.append(dependent[i])
        result = 0.0
        for i in range(len(independent_test)):
            calculated = theta0+(theta1*independent_test[i])
            result += (dependent_test[i]-calculated)**2
    error = result/len(independent_test)
    print("Mean Squared Error = ",error)
    print("The Accuracy of you trained Model is = ", round(100-error,2),'%')
print("Model is ready for calculations!")
while i!= 'q':
    x = input("Enter independent value to predict OR 's' to show the theta0 and theta1 values OR 'q' to quit = ")
    if x == 'q':
        break
    if x == 's':
        print(f'Y = {theta0} + {theta1} X')
        break
    x = float(x)
    y = theta0+(theta1*x)
    print("Predicted value is = ",y)