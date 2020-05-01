import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import random
from math import sqrt
import pymongo
from pymongo import MongoClient
import json
import time

def data():
    connection = MongoClient('localhost', 27017)
    db = connection.recsys
    df = pd.read_csv('C:/Users/HI/Documents/internship recommendation system/ml-25m/ratings.csv')
    movies = pd.read_csv('C:/Users/HI/Documents/internship recommendation system/ml-25m/movies.csv')
    df = pd.DataFrame(list(ratings.find()))
    movies = pd.DataFrame(list(movies.find()))
    new_df = pd.concat([movies.drop('genres', 1), movies['genres'].str.get_dummies(sep="|")], 1)
    new_df.drop(new_df[new_df['(no genres listed)']==1].index,inplace=True)
    new_df = new_df.drop(['(no genres listed)'], axis = 1)
    df = pd.merge(df, new_df, on='movieId')
    print('data')
    return df

def euclidean(u1,u2,weights,df):
    s = pd.merge(df[df['userId']==u1], df[df['userId']==u2], how="inner", on="movieId")
    sum1 = 0
    similarity=0
    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]
    w4 = weights[3]
    w5 = weights[4]
    w6 = weights[5]
    w7 = weights[6]
    w8 = weights[7]
    w9 = weights[8]
    w10 = weights[9]
    w11 = weights[10]
    w12 = weights[11]
    w13 = weights[12]
    w14 = weights[13]
    w15 = weights[14]
    w16 = weights[15]
    w17 = weights[16]
    w18 = weights[17]
    w19 = weights[18]
    w20 = weights[19]
    
    for index, row in s.iterrows():
           
        tempvar = (w1*((row.rating_x - row.rating_y)**2) + \
                w2*((row.Action_x - row.Action_y)**2) + \
                w3*((row.Adventure_x - row.Adventure_y)**2) + \
                w4*((row.Animation_x - row.Animation_y)**2) + \
                w5*((row.Children_x - row.Children_y)**2) + \
                w6*((row.Comedy_x - row.Comedy_y)**2) + \
                w7*((row.Crime_x - row.Crime_y)**2) + \
                w8*((row.Documentary_x - row.Documentary_y)**2) + \
                w9*((row.Drama_x - row.Drama_y)**2) + \
                w10*((row.Fantasy_x - row.Fantasy_y)**2) + \
                w11*((row.SciFi_x-row.SciFi_y)**2) + \
                w12*((row.Horror_x - row.Horror_y)**2) + \
                w13*((row.IMAX_x - row.IMAX_y)**2) + \
                w14*((row.Musical_x - row.Musical_y)**2) + \
                w15*((row.Mystery_x - row.Mystery_y)**2) + \
                w16*((row.Romance_x - row.Romance_y)**2) + \
                w17*((row.FilmNoir_x-row.FilmNoir_y)**2) + \
                w18*((row.Thriller_x - row.Thriller_y)**2) + \
                w19*((row.War_x - row.War_y)**2) + \
                w20*((row.Western_x - row.Western_y)**2)    
                  ) 
        tempvar = max(0, tempvar)
        tempvar = math.sqrt(tempvar)
        sum1 += tempvar
        distance = sum1/s.shape[0]
        similarity = 1/(1+distance)
        #print('eucledian distance')
        
    return similarity

def similarUsers(user,user_weight_dict,df):
    k = 25
    sim_scores = []
    for i in df.userId.unique():
        s = euclidean(user, i, user_weight_dict,df)
        #s = correlation(user,i)
        #s = euclidean_similarity(user,i)
        sim_scores.append(s)
    simusers = {'users':df.userId.unique(), 'similarity_score':sim_scores}
    simusers = pd.DataFrame(simusers)
    simusers = simusers.nlargest(k, 'similarity_score')
    #print('similar users')
    return simusers

def predictRating(user,movie,similar_user,df):
    n = 0
    predRating = 0
    k = 25
    #s = similarUsers(user)
    for i in similar_user['users']:
        rating = df[(df['userId']==i) & (df['movieId'] == movie)]['rating'].values
        sim = similar_user[similar_user['users']==i]['similarity_score'].values
        if(len(rating)!=0):
            predRating = predRating + (rating * sim)
    predRating = predRating/k 
    rating = df[df['userId']==user]['rating'].values
    average = np.average(rating)
    predRating = average+predRating
    if(predRating>5):
        predRating = 5
    #print('pr')
    return predRating

def getTopMovies1(user,similar_user,df):
    top_movies = []
    #s = similarUsers(testUser)
    for i in similar_user['users']:
        if i == user:
            continue
        movies = df[(df['userId']==i) & (df['rating']>4)]['movieId'].values
        for j in movies:
            if j not in top_movies:
                top_movies.append(j)
    #print('movies')
    return top_movies

def recommendation1(testUser,similar_user,df):
    mo = []
    p = []
    u = []
    m = getTopMovies1(testUser,similar_user,df)
    pr = 50
    if(len(m)<50):
        pr = len(m)
    for j in range(0,pr):
        predRating = predictRating(testUser,m[j],similar_user,df)
        mo.append(m[j])
        u.append(testUser)
        if np.isscalar(predRating):
            p.append(predRating)
        else:
            p.append(predRating[0])
    d = {'userId':u,'movieId':mo, 'predicted rating': p}
    df_u = pd.DataFrame(d)
    df_u = df_u.nlargest(50, 'predicted rating')
    #print('rec')
    return df_u

def fitness_func(weights,testUser,df):
    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]
    w4 = weights[3]
    w5 = weights[4]
    w6 = weights[5]
    w7 = weights[6]
    w8 = weights[7]
    w9 = weights[8]
    w10 = weights[9]
    w11 = weights[10]
    w12 = weights[11]
    w13 = weights[12]
    w14 = weights[13]
    w15 = weights[14]
    w16 = weights[15]
    w17 = weights[16]
    w18 = weights[17]
    w19 = weights[18]
    w20 = weights[19]
    
    w_sum = w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12+w13+w14+w15+w16+w17+w18+w19+w20
    
    w1 = w1/w_sum
    w2 = w2/w_sum
    w3 = w3/w_sum
    w4 = w4/w_sum
    w5 = w5/w_sum
    w6 = w6/w_sum
    w7 = w7/w_sum
    w8 = w8/w_sum
    w9 = w9/w_sum
    w10 = w10/w_sum
    w11 = w11/w_sum
    w12 = w12/w_sum
    w13 = w13/w_sum
    w14 = w14/w_sum
    w15 = w15/w_sum
    w16 = w16/w_sum
    w17 = w17/w_sum
    w18 = w18/w_sum
    w19 = w19/w_sum
    w20 = w20/w_sum
    
    w = [w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20]
    similar_user = similarUsers(testUser,w,df)
    df_u = recommendation1(testUser,similar_user,df)
    df_err = pd.merge(df,df_u,on=['movieId','userId'])
    if(df_err.shape[0]<1):
        return 100000
    rmse = sqrt(mean_squared_error(df_err['rating'], df_err['predicted rating']))
    print('fitness')
    return rmse

lb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
ub = [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]

# implementation of PSO
# checking and defining weight inside the function; a random number between 0.5 and 1
def pso(fitness_func, lbound, ubound, u, df, swarm_size=2, max_iter=1, dimensions=20, c1=1.494, c2=1.494):
    x=[]            # positon of each particle
    x_pbest=[]      # best position of each particle 
    x_gbest=[]      # best global position of population
    v=[]            # velocity of each particle
    v_min=[]
    v_max=[]
    
    # assign initial random positions to the particles
    for i in range(swarm_size):
        x.append([])
        x_pbest.append([])
        for j in range(dimensions):
            x[i].append(random.uniform(lbound[j],ubound[j]))
            x_pbest[i].append(x[i][j])
            
    #calculate initial group best of the population
    for i in range(swarm_size):
        if i==0:
            x_gbest=x[i]
        elif fitness_func(x[i],u,df)<fitness_func(x_gbest,u,df):
            x_gbest=x[i]

    
    # calculate minimum and maximum boundaries of velocity vector
    for i in range(dimensions):
        v_min.append(-(ubound[i]-lbound[i])/100)
        v_max.append((ubound[i]-lbound[i])/100)
 
    # assign initial random velocities to the particles
    for i in range(swarm_size):
        v.append([])
        for j in range(dimensions):
            v[i].append(random.uniform(v_min[j],v_max[j]))
            
    for iter in range(max_iter):
        for i in range(swarm_size):
            for j in range(dimensions):
                r1=random.uniform(0,1)
                r2=random.uniform(0,1)
                weight = random.uniform(0.5, 1)
                # calculate new velocity for each particle
                v[i][j] = weight*(v[i][j]) + r1*c1*(x_pbest[i][j]-x[i][j]) + r2*c2*(x_gbest[j]-x[i][j])
                
                if v[i][j] > v_max[j]:
                    v[i][j] = v_max[j]
                
                if v[i][j] < v_min[j]:
                    v[i][j] = v_min[j]
                                 
                # calculate new position for each particle
                x[i][j] = x[i][j] + v[i][j]
                
                if x[i][j] > ubound[j]:
                    x[i][j] = ubound[j]
                    
                if x[i][j] < lbound[j]:
                    x[i][j] = lbound[j]
            
            if fitness_func(x[i],u,df) < fitness_func(x_pbest[i],u,df):
                x_pbest[i] = x[i]
                
            if fitness_func(x[i],u,df) < fitness_func(x_gbest,u,df):
                x_gbest = x[i]
                
    return x_gbest,fitness_func(x_gbest,u,df)

def recommendation(testUser,similar_user):
    mo = []
    p = []
    u = []
    m = getTopMovies(testUser,df)
    pr = 50
    if(len(m)<50):
        pr = len(m)
    for j in range(0,pr):
        predRating = predictRating(testUser,m[j],df)
        mo.append(m[j])
        u.append(testUser)
        #print(j)
        if np.isscalar(predRating):
            p.append(predRating)
        else:
            p.append(predRating[0])
    d = {'userId':u,'movieId':mo, 'predicted rating': p}
    df_u = pd.DataFrame(d)
    df_u = df_u.nlargest(50, 'predicted rating') 
    return df_u

def main():
    df = data()
    df_c = pd.DataFrame(columns=['userId','movieId','predicted rating'])
    for testUser in df.userId.unique():
        print(testUser)
        xopt, fopt = pso(fitness_func, lb, ub,testUser,df)
        print('pso done')
        w = []
        for i in xopt:
            w.append(i/sum(xopt))
        similar_user = similarUsers(testUser,w,df)
        df_u = recommendation1(testUser,similar_user,df)
        df_c = pd.concat([df_u, df_c])
        
        print('recommendations for user', testUser, ':\n')
        l = 10
        if(len(df_u)<10):
            l = len(df_u['movieId'])
        for i in range(0,l):
            m = df[df['movieId']==df_u['movieId'][i]]['title'].values
            print(m[0])
        print('\n\n')
        rec = rec+m[0]
    rec = json.load(rec)

if __name__ == '__main__':
    main()
