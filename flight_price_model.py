import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 

###EDA
train_data=pd.read_excel(r"E:\DS Project\Flight-Price-Prediction-master\Data_Train.xlsx")
pd.set_option('display.max_columns', None)
train_data.head()
train_data.info()
train_data.shape
train_data.isnull().sum()
train_data["Airline"].value_counts()
sns.catplot(y = "Price", x = "Airline", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()
###########
train_data["Source"].value_counts()
sns.catplot(y = "Price", x = "Source", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)
plt.show()
train_data["Destination"].value_counts()
sns.catplot(y = "Price", x = "Destination", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)
plt.show()
train_data["Route"]
train_data["Additional_Info"]

#train_data.dropna(inplace = True)
class feature_engineering(BaseEstimator, TransformerMixin):
    
    
        
        
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X.dropna(inplace = True)
        if "Date_of_Journey" in X.columns:
            X['Journey_day']= pd.to_datetime(X['Date_of_Journey'], format="%d/%m/%Y").dt.day
            X['Journey_month'] = pd.to_datetime(X['Date_of_Journey'], format = "%d/%m/%Y").dt.month
            X.drop(["Date_of_Journey"], axis = 1, inplace = True)
        
        #############################
        a=("Dep_hour","Arrival_hour")
        b=("Dep_Time","Arrival_Time")
        c=("Dep_min","Arrival_min")
        for i,j,k in zip(a,b,c):
             if j in X.columns:
                 X[i] = pd.to_datetime(X[j]).dt.hour
                 X[k] = pd.to_datetime(X[j]).dt.minute
        ####################
                 X.drop([j], axis = 1, inplace = True)
             else:
                 pass
    #################
        if  'Duration' in  X.columns: 
            duration_hours = []
            duration_mins = [] 
            duration = list(X["Duration"])
            for i in range(len(duration)):
            
               for i in range(len(duration)):
                  if len(duration[i].split()) != 2:
                      if "h" in duration[i]:
                        duration[i] = duration[i].strip()+ " 0m"
                      else:
                        duration[i] = "0h " + duration[i]
            for i in range(len(duration)):
                duration_hours.append(int(duration[i].split(sep = "h")[0])) 
                duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))
                #####################
            X["Duration_hours"] = duration_hours
            X["Duration_mins"] = duration_mins   
            X.drop(["Duration"], axis = 1, inplace = True)
                  
                

        return X
#x1=feature_engineering()
#trains_d=x1.transform(train_data)   

#Destination = train_data[["Destination"]]  
 
#########
#handle ctagorical feature and delete unwanted categorical columns
class cat_encoder(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        
        for i in ["Airline","Source","Destination"]:
            a=X[[i]]
            v=pd.get_dummies(a, drop_first= True)
            X=pd.concat([X,v],axis = 1)
            X.drop([i], axis = 1, inplace = True)
        X.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
        X.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
        return X        

#x1=cat_encoder()   
#trains_d=x1.transform(train_data)       
#x.isnull().sum()      
#trains_data.columns            
#trains_data.shape
class splitter(BaseEstimator, TransformerMixin):
    def fit(self,data):
        return self
    def transform(self,data):
        X=data.drop("Price",axis=1)
        y=data["Price"]
        return X,y
#x1= splitter()
#x2,y1=x1.transform(train_data)
    
my_pipeline=Pipeline([("feat_engi",feature_engineering()),
                      ("cat_encoder",cat_encoder()),
                       ("splitter",splitter())])
x2,y2=my_pipeline.fit_transform(train_data)
#x2.shape
    
#######################
a=['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins']
x3=x2.loc[:,a]
x3=pd.concat([x3,y2],axis=1)

# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (18,18))
sns.heatmap(x3.corr(), annot = True, cmap = "RdYlGn")

plt.show()
    
from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor(random_state=0)
selection.fit(x2,y2)  
##############
print(selection.feature_importances_)
########
plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=x2.columns)
feat_importances.nlargest(20).plot(kind='barh')
feat_importances.nlargest(20).index

plt.show()

feature=['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers', 'Airline_Vistara','Source_Delhi',
       'Source_Mumbai','Destination_Delhi', 'Destination_Hyderabad',
       'Destination_New Delhi']

class feature_selector(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        X=X.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers', 'Airline_Vistara','Source_Delhi',
       'Source_Mumbai','Destination_Delhi', 'Destination_Hyderabad',
       'Destination_New Delhi']]
        
        return X.values
    
################   
train_data.dropna(inplace = True)
X=train_data.drop("Price",axis=1)
y=train_data["Price"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)       

 ##########   
my_pipeline2=my_pipeline=Pipeline([("feat_engi",feature_engineering()),
                      ("cat_encoder",cat_encoder()),
                       ("Feat_selector",feature_selector())])

X_train1=my_pipeline2.fit_transform(X_train)
X_test1=my_pipeline2.transform(X_test)
y_train= y_train.values
y_test=y_test.values 


########
#from sklearn.ensemble import RandomForestRegressor

#my_fullPipeline=Pipeline([("my_pipe",my_pipeline2),
#                          ("model",RandomForestRegressor())])

#my_fullPipeline.fit(X_train,y_train)

#y_pred=my_fullPipeline.predict(X_test)

############
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

pipeline_lr=Pipeline([("Scaler",StandardScaler()),
                      ("Model",LinearRegression())])

pipeline_dc=Pipeline([("Decision_tree",DecisionTreeRegressor())])

pipeline_rm=Pipeline([("random_forest",RandomForestRegressor(random_state=0))])

#my_fullPipeline=Pipeline([("my_pipe",my_pipeline2),
#                              ("model",pipeline_lr)])

#my_fullPipeline.fit(X_train,y_train)
#my_fullPipeline.score(X_test,y_test)

piplines=[pipeline_lr,pipeline_dc,pipeline_rm]

best_score=0.0
best_regresssor=0
best_pipeline=" "

pipe_dict={0:"linear regression",1:"Decision Tree",2:"Random Forest"}

for i,model in enumerate(piplines):
    
    my_fullPipeline=Pipeline([("my_pipe",my_pipeline2),
                              ("model",model)])
    my_fullPipeline.fit(X_train,y_train)
    if my_fullPipeline.score(X_test,y_test)>best_score:
        best_score=my_fullPipeline.score(X_test,y_test)
        best_regresssor=i
        best_pipeline=piplines[i]
print("best model:{} ,score:{},beat pipeline:{}".format(pipe_dict[i],
                                                        best_score,best_pipeline))    


#########################

from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train1, y_train)
##########
y_pred = reg_rf.predict(X_test1)
reg_rf.score(X_test1, y_test)
#############
sns.distplot(y_test-y_pred)
plt.show()
############
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
###########
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
############
metrics.r2_score(y_test, y_pred)
##########
from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf}
####################
#rf_random = RandomizedSearchCV(estimator = reg_rf,
 #                              param_distributions = random_grid,
#                               scoring='neg_mean_squared_error',
#                               n_iter = 10, cv = 5, verbose=2,
#                               random_state=42, n_jobs = 1)
#rf_random.fit(X_train1,y_train)

#############
#rf_random.best_params_

######################################

rf_random=RandomForestRegressor(n_estimators= 700,min_samples_split= 15,
                                min_samples_leaf= 1,max_features= 'auto',
                                max_depth= 20)
rf_random.fit(X_train1,y_train)
pred = rf_random.predict(X_test1)
metrics.r2_score(y_test,pred)
##########
plt.figure(figsize = (8,8))
sns.distplot(y_test-pred)
plt.show()
##########
plt.figure(figsize = (8,8))
plt.scatter(y_test, pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
########
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

##########


my_fullPipeline=Pipeline([("my_pipe",my_pipeline2),
                         ("model",rf_random)])

my_fullPipeline.fit(X_train,y_train)
my_fullPipeline.score(X_test,y_test)


###################
import joblib

file = open('model.pkl','wb')
joblib.dump(rf_random, file)
file.close()

#############
model = open('model.pkl','rb')
forest = joblib.load(model)
model.close()
######
y_prediction = forest.predict(X_test1)
metrics.r2_score(y_test, y_prediction)

