import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import linregress
from sklearn.model_selection import train_test_split



url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

#===========================================================
# fixing data
#============================================================
# checking how many null values -> 77% deleting column
print((df['deck'].isnull().sum() / len(df)) *100)
df = df.drop(columns='deck')

# checking the same for age  -> 19%, filling with the mean age
print(df['age'].isnull().sum()/len(df)*100)
df['age'] = np.where(df['age'].isnull(), df['age'].mean(), df['age']).round()

# change: male: 0 , female: 1
df['sex'] = np.where(df['sex'] == 'female', 1, 0)






# =============================================
# KNN algo
# =============================================

# distance func
def dist(v1,v2):
    return np.sqrt(np.sum((v1-v2)**2))

# a class of KNN regression
class KNNregressor():

    def __init__(self, features, res_vec, k = 3):
        self.k = k
        self._res_vec = np.array(res_vec)
        self._features = np.array(features)

    def K_close_list(self,pt):
        # creating min heap (the class min_heap is below)
        heap = min_heap()

        # for every point(human) in the data(which is a row in the features)
        # calculating the distances and adding the tuple (distance from pt, survvival result)
        for i, train_point in enumerate(self._features):

            d = dist(pt, train_point)
            heap.insert_to_heap((d, self._res_vec[i]))

        # for K times extracting the min tuple from the min heap to the list
        # and then finally returning the list of tuples
        k_closest = []
        for _ in range(self.k):
            tup = heap.pop_min()
            if tup is None:
                break
            k_closest.append(tup)
        return k_closest
    # returning predicted probability of survival
    def predict(self, pt):
        neighbors = self.K_close_list(pt)
        # creating a list of only the survival(0,1) result without the distances
        # and then taking the average
        survive_res_list = [res[1] for res in neighbors]
        return np.sum(survive_res_list)/self.k
    


#the same as KNNregressor except here the prediction is weighted
class KNNimproved(KNNregressor):
    def __init__(self, features, res_vec, k=3):
        super().__init__(features, res_vec, k)
    
    def predict(self, pt):

        neighbors = self.K_close_list(pt)
        
        weighted_sum = 0
        total_weights = 0
        epsilon = 1e-5 
        #seperating the distances and survival result to different variables
        for distance, survive_res in neighbors:
           # if the distance is close it has more weight than far points
           # we need the epsilon in case there is a point with distance 0(the same place as pt)
            weight = 1 / (distance + epsilon)
            weighted_sum += weight * survive_res
            total_weights += weight
        # retunrning the prediction 
        return weighted_sum / total_weights
    


# creating a min heap for maybe online use in the future
class min_heap():
    def __init__(self):
        self.heap_list = []


    def insert_to_heap(self, item):
        # adding the item and tracking its index
        self.heap_list.append(item)
        item_idx = len(self.heap_list) - 1
        # entering if there's more than one item in the heap
        while item_idx > 0:
            # finding father index
            father_idx = (item_idx-1) //2
            #if it's father is bigger than his than the item we 
            # added should be replaces with him
            if self.heap_list[father_idx] > self.heap_list[item_idx]:
                # tuple switch
                self.heap_list[father_idx], self.heap_list[item_idx] = self.heap_list[item_idx], self.heap_list[father_idx]
                item_idx = father_idx
            else:
                break

    def pop_min(self):
        # if there're no items in the heap
        if len(self.heap_list)==0:
            return None 
        # taking out the last value in the list
        min_val =self.heap_list[0]
        last_val= self.heap_list.pop()
        # now if the list is length 0 after taking out the idex
        # then there was one item and we just need to retun it
        if len(self.heap_list) == 0:
            return min_val

        #if that's not the case we need to fix the heap before returning the min
        # putting the last index instead of the root
        self.heap_list[0] = last_val
        item_idx = 0
        length = len(self.heap_list)

        while True:
            # finding the childern of the node we comaring
            left_child_idx = 2 * item_idx + 1
            right_child_idx = 2 * item_idx + 2
            smallest_idx = item_idx
            # if the left child in the list is smaller than the value if the current min index
            # then update who's the smallest index
            if left_child_idx < length and self.heap_list[left_child_idx]< self.heap_list[smallest_idx]:
                smallest_idx=left_child_idx
            # the same as the left
            if right_child_idx < length and self.heap_list[ right_child_idx ]< self.heap_list[smallest_idx]:
                smallest_idx=right_child_idx
            # if the smallest index is not the father than we should replace
            # him with the smallest child
            if smallest_idx!=item_idx:
                self.heap_list[item_idx], self.heap_list[smallest_idx] = self.heap_list[smallest_idx], self.heap_list[item_idx]
                # update the father index
                item_idx = smallest_idx
            else:
                # the father is smaller than his kids we're done 
                break
        # fixed the heap now returning the min value
        return  min_val
    


# =======================================================
# some graphs
# =======================================================


fig, ax = plt.subplots(2, 1, figsize=(6, 6))

# --- Graph 1: Survival by Sex ---
sex_survival = df.groupby('sex')['survived'].mean()

ax[0].bar(['Male', 'Female'], sex_survival.values, color=['blue', 'red'])
ax[0].set_title('Survival Rate by Sex')   
ax[0].set_ylabel('Chance to Survive')        

# --- Graph 2: Survival by Class ---
class_survival = df.groupby('pclass')['survived'].mean()
ax[1].bar(['First', 'Second', 'Third'], class_survival.values, width=0.4, color='green')
ax[1].set_title('Survival Rate by Class')
ax[1].set_ylabel('Chance to Survive')

plt.tight_layout()
plt.show()
plt.close()


age_data = df.groupby('age')['survived'].mean()
x = age_data.index.values   
y = age_data.values

x_graph = np.linspace(x.min(), x.max(), 100)
# degree 3 is good because 4 is overfitting :)
coeffs = np.polyfit(x, y, deg=3)
y_graph = np.polyval(coeffs, x_graph)
plt.scatter(x, y,s = 4, alpha = 0.6, label = "data points")
plt.grid(True)
plt.ylabel("chance to survive (%)")
plt.xlabel("age")
plt.plot(x_graph, y_graph, color='red', label = "poly regress")

# X - random variable of age
def survival_integral(X):
    return np.polyval(coeffs, X)

res, error = quad(survival_integral, x.min(), x.max())
plt.scatter([], [], label=f'Integrated Survival Probability: {res:.2f}', color = 'white')
plt.legend() 
plt.show()
plt.close()


# ======================================================================
# KNN algo
# =====================================================================

"""
the goal of the experiment:
1) to see how accurate KNN can be in assessing to probability of surviving giving a new data point
2) to comapre the regular to the weighted KNN that I've built to see which one works better
3) to see what value of k(how many neighbors we check) will give the best result 

haypothesis:
the weighted KNN will be better and and k = 5 will be the best(just saying on gut feeling lol)
"""

# Select only the relevant features for the model
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
# X the matrix (every human is a row), Y is the vector of survival result
X = df[features]
Y = df['survived']

k_vals_check = range(1,20,2)

X_norm = (X - X.mean()) / X.std()
X_train, X_test, Y_train, Y_test = train_test_split(X_norm.values, Y.values, test_size=0.2, random_state=42)

reg_KNN = []
impv_KNN = []

for k in k_vals_check:
    model_reg = KNNregressor(X_train, Y_train, k)
    model_impv = KNNimproved(X_train, Y_train, k)
    
    correct_reg = 0
    correct_impv = 0
    
    for i in range(len(X_test)):
        pt = X_test[i]
        true_val = Y_test[i]
        
        pred_reg = 1 if model_reg.predict(pt) > 0.5 else 0
        if pred_reg == true_val:
            correct_reg += 1
            
        pred_impv = 1 if model_impv.predict(pt) > 0.5 else 0
        if pred_impv == true_val:
            correct_impv += 1
            
    reg_KNN.append(correct_reg / len(X_test))
    impv_KNN.append(correct_impv / len(X_test))

plt.figure(figsize=(10, 6))
plt.plot(k_vals_check, reg_KNN, label='Regular KNN', marker='o')
plt.plot(k_vals_check, impv_KNN, label='Improved KNN', marker='s')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


"""
findings:

it seems like the accuracy of them both is about the same with maybe a neglegable 
improvement with the regular KNN, it also seems like between 7-10 is the best k
(probably 9 according to the chart)

"""