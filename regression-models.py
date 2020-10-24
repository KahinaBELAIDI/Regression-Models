#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#load the dataset
df = pd.read_csv("kc_house_data.csv")

df.describe()

df.info()

df.isnull().sum()


#select main features
"""
- Define the plot 
- Create a heatmap
"""
corr= df.corr()
figure, ax = plt.subplots(figsize=(15,8))
sns.heatmap(corr,annot=True,cmap="Purples" )


new_df= df[["sqft_living", "grade","sqft_above","sqft_living15","bathrooms", "price"]]

#split data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X1 = df["sqft_living15"].values.reshape(-1,1)
Y = df["price"]
x_train, x_test, y_train, y_test = train_test_split(X1,Y, test_size=0.2, random_state=5)


#print("\n x_train: \n", x_train)
#print("\n y_train : \n", y_train)
#print("\n x_test : \n", x_test)
#print("\n y_test : \n", y_test)



model = LinearRegression()
model.fit(x_train,y_train)


coeff =  model.score(x_train, y_train)
print("\n coeff : ", coeff)


prediction = model.predict(x_test)
print("\n prediction :", prediction )


from sklearn import metrics
mse = metrics.mean_squared_error(y_test, prediction)
print("\n mse = ", mse)
print("\n model.coef_ : ",model.coef_)
r2 = metrics.r2_score(y_test,prediction)
print("\n r2 : ", r2)

plt.plot(X1,model.predict(X1),color="red")
plt.scatter(X1,Y,color="blue")
plt.title("Linear Regression")
plt.ylabel("price")
plt.xlabel("sqft_living15")
plt.show()

#multilinear regression
x_multi = df[["sqft_living", "grade","sqft_above","sqft_living15","bathrooms"]]
Y = df["price"]
x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(x_multi,Y, test_size=0.2, random_state=5)
multi_model = LinearRegression()
multi_model.fit(x_train_m, y_train_m)
multi_predict = multi_model.predict(x_test_m)
print("/n multi predict : ", multi_predict)

plt.plot(x_multi, multi_model.predict(x_multi), color='green')
#plt.scatter(x_multi, Y, color="yellow")
plt.show()

r2_multi = metrics.r2_score(y_test_m,multi_predict)


print("\n model.coef_ : ",multi_model.coef_)


rmse = (np.sqrt(metrics.mean_squared_error(y_test_m,multi_predict)))

#Polynomial regression

poly = PolynomialFeatures(degree = 3) 

x_train_poly=poly.fit_transform(x_train_m)
multi_model.fit(x_train_poly, y_train_m)


x_test_poly=poly.fit_transform(x_test_m)
predicted_poly= multi_model.predict(x_test_poly)

print("MSE for poly regression: ", metrics.mean_squared_error(y_test_m, predicted_poly))
print("R squared for poly regression: ", metrics.r2_score(y_test_m,predicted_poly),"\n")

accurcy3=multi_model.score(x_test_poly,y_test_m)
print("accuracy : ",round(accurcy3*100),"%")


