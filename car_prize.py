import pandas  as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
df=pd.read_csv("car_data.csv")
df=df.drop("Car_Name",axis=1)
df=pd.get_dummies(df,drop_first=True)
x=df.drop("Selling_Price",axis=1)
y=df["Selling_Price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestRegressor(random_state=42)
model.fit(x_train,y_train)
predictions=model.predict(x_test)
score=r2_score(y_test,predictions)
print("r2 Score",score)
pickle.dump(model,open("Car_model","wb"))
print("model save sucessfully")
year=int(input("enter year:"))
present_price=float(input("Enter present price: "))
driven_kms=int(input("enter driven kms: "))
owner=int(input("Enter owner count: "))
fuel_disel=int(input("Fuel Diesel?(1/0): "))
fuel_petrol=int(input("fuell petrol?(1/0): "))
selling_indivisual=int(input("Individual Seller?(1/0): "))
transmission_manual=int(input("Manual Transmission?(1/0): "))
user_data=pd.DataFrame([[year,
           present_price,
           driven_kms,
           owner,
           fuel_disel,
           fuel_petrol,
           selling_indivisual,
           transmission_manual
           ]],
       columns=[
    'Year',
    'Present_Price',
    'Driven_kms',
    'Owner',
    'Fuel_Type_Diesel',
    'Fuel_Type_Petrol',
    'Selling_type_Individual',
    'Transmission_Manual'
])
        

predicted_price=model.predict(user_data)
print("\n predicted car price=",predicted_price[0],"lakhs")